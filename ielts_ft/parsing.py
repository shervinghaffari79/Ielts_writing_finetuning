"""Turn the raw spreadsheet text (Rater2_* columns + Feedback) into the structured target.

Why this is needed: the spreadsheet stores feedback as free text, but the model must learn a strict
JSON format. Along the way we enforce the rules from the Stage 1 objectives document, above all
"evidence must be short, exact quotations from the essay" - an audit of the source data showed
that only ~44% of the quoted 'examples' are verbatim (the rest are corrected/paraphrased), so every
quotation is verified against the essay and either kept, snapped to the real wording, or dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .schema import (CRITERIA, CRITERION_TITLES, MIN_WORDS, count_words, fmt_band, normalize_text,
                     quote_in_essay)

MAX_QUOTE_WORDS = 22
MAX_ITEMS = 4          # strengths / weaknesses kept per criterion
MAX_EVIDENCE = 5

_BAND_RE = re.compile(r"Band\s*([0-9](?:\.[05])?)", re.I)
_POS_RE = re.compile(r"Positives?\s*:", re.I)
_NEG_RE = re.compile(r"Negatives?\s*:", re.I)
_QUOTE_RE = re.compile(r'["“]([^"“”]{2,}?)["”]')
_BULLET_RE = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s*")
_HEADING_RE = re.compile(r"^(?:[A-Z][A-Za-z/&'\- ]{2,45}):\s+(?=\S)")
_CORRECTION_GAP = re.compile(r"(?:should|could|would|can|might)\s+(?:be|better|have)|better\s+(?:as|phrased|expressed)|"
                             r"corrected?\s+(?:to|as)|instead\s+of|rather\s+than|replaced?\s+(?:with|by)|changed?\s+to|"
                             r"\bwrite\b|\buse\b", re.I)
_CORRECTION_CUT = re.compile(r"\s*,?\s*(?:which|that|this)?\s*(?:should|could|can|would|might)\s+(?:be|better|have)\b.*$",
                             re.I | re.S)


# ----------------------------------------------------------------------------- quote helpers
def trim_quote(q: str, max_words: int = MAX_QUOTE_WORDS) -> str:
    q = q.strip().strip("\"“”").strip()
    w = q.split()
    return " ".join(w[:max_words]) if len(w) > max_words else q


def snap_quote(quote: str, essay: str, threshold: float = 90.0) -> str | None:
    """Find the essay wording closest to `quote` (rapidfuzz over word windows)."""
    qn = normalize_text(quote)
    nq = len(qn.split())
    if nq < 2 or "..." in quote or "…" in quote:
        return None
    words = essay.split()
    best, best_s = None, 0.0
    for n in range(max(1, nq - 1), nq + 2):
        for i in range(0, max(1, len(words) - n + 1)):
            cand = " ".join(words[i:i + n])
            s = fuzz.ratio(qn, normalize_text(cand))
            if s > best_s:
                best, best_s = cand, s
    return best if best_s >= threshold else None


def resolve_quote(q: str, essay: str) -> str | None:
    """Return the verbatim essay wording for a quote, or None if it cannot be grounded."""
    q = q.strip().strip("\"“”").strip()
    if quote_in_essay(q, essay):
        return trim_quote(q)
    s = snap_quote(q, essay)
    return trim_quote(s) if s else None


# ----------------------------------------------------------------------------- sentence handling
def _protect(text: str):
    store: list[str] = []

    def rep(m):
        store.append(m.group(0))
        return f"⟦{len(store) - 1}⟧"

    return _QUOTE_RE.sub(rep, text), store


def _restore(text: str, store: list[str]) -> str:
    return re.sub(r"⟦(\d+)⟧", lambda m: store[int(m.group(1))], text)


def split_sentences(block: str) -> list[str]:
    """Split a paragraph into sentences without breaking inside quotation marks."""
    prot, store = _protect(block)
    out = []
    for line in re.split(r"\n+", prot):
        line = _BULLET_RE.sub("", line).strip().replace("**", "")
        if not line:
            continue
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z⟦])", line):
            s = s.strip()
            if s:
                out.append(_restore(s, store))
    return out


@dataclass
class ParsedCriterion:
    band: float | None = None
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    weak_quotes: list[str] = field(default_factory=list)     # verified quotes cited as problems
    strong_quotes: list[str] = field(default_factory=list)   # verified quotes cited as positives
    stats: dict = field(default_factory=lambda: dict(quotes=0, verbatim=0, snapped=0, dropped=0,
                                                       sentences_dropped=0))


def _process_sentence(sent: str, essay: str, stats: dict, evidence: list[str]) -> str | None:
    # `evidence` receives every verified quote found in the sentence (in order)
    """Ground every quotation in `sent`. Returns the cleaned sentence or None to drop it."""
    sent = _HEADING_RE.sub("", sent).strip()
    if re.match(r"^(?:for\s+)?examples?\b[^:]*:", sent, re.I):
        sent = re.sub(r"^(?:for\s+)?examples?\b[^:]*:\s*", "", sent, flags=re.I)
        if not sent:
            return None
    spans = list(_QUOTE_RE.finditer(sent))
    out, last, first_bad = [], 0, None
    for m in spans:
        if out and _CORRECTION_GAP.search(sent[last:m.start()]):
            # a quotation introduced by "should be / instead of ..." is a correction, not the student's text
            first_bad = last
            break
        stats["quotes"] += 1
        raw = m.group(1)
        exact = quote_in_essay(raw, essay)
        real = trim_quote(raw) if exact else (snap_quote(raw, essay) and trim_quote(snap_quote(raw, essay)))
        if real:
            stats["verbatim" if exact else "snapped"] += 1
            if real not in evidence:
                evidence.append(real)
            out.append(sent[last:m.start()] + '"' + real + '"')
            last = m.end()
        else:
            stats["dropped"] += 1
            first_bad = m.start()
            break
    if first_bad is not None:
        if not out:
            stats["sentences_dropped"] += 1
            return None
        # keep the text up to the last verified quote; drop the "...should be '<correction>'" tail
        sent = "".join(out)
        sent = _CORRECTION_CUT.sub("", sent).rstrip(" ,;:")
    else:
        sent = "".join(out) + sent[last:]
    sent = re.sub(r"\s+", " ", sent).strip()
    if re.fullmatch(r'["\u201c][^"]*["\u201d][.,;]?', sent) or len(sent.split()) < 4:
        return None   # a bare quotation is evidence, not a strength/weakness statement
    if sent[-1] not in ".!?\"":
        sent += "."
    return sent


def parse_criterion_text(text: str, essay: str) -> ParsedCriterion:
    pc = ParsedCriterion()
    m = _BAND_RE.search(text)
    pc.band = float(m.group(1)) if m else None
    pm, nm = _POS_RE.search(text), _NEG_RE.search(text)
    if not (pm and nm):
        return pc
    if pm.start() < nm.start():
        pos_text, neg_text = text[pm.end():nm.start()], text[nm.end():]
    else:
        neg_text, pos_text = text[nm.end():pm.start()], text[pm.end():]
    for block, dest, qs in ((pos_text, pc.strengths, pc.strong_quotes), (neg_text, pc.weaknesses, pc.weak_quotes)):
        for sent in split_sentences(block):
            cleaned = _process_sentence(sent, essay, pc.stats, qs)
            if cleaned and cleaned not in dest:
                dest.append(cleaned)
    pc.strengths, pc.weaknesses = pc.strengths[:MAX_ITEMS], pc.weaknesses[:MAX_ITEMS]
    # problems first: they are what the student has to work on
    for q in pc.weak_quotes + pc.strong_quotes:
        if q not in pc.evidence:
            pc.evidence.append(q)
    pc.evidence = pc.evidence[:MAX_EVIDENCE]
    return pc


# ----------------------------------------------------------------------------- Feedback column
_OVERALL_RE = re.compile(r"^(Task Response|Coherence and Cohesion|Lexical Resource|"
                         r"Grammatical Range and Accuracy)\s*:\s*(.+)$", re.M)
_IMPROVE_RE = re.compile(r"^Improvement point:\s*(.+)$", re.M)
_TITLE_TO_KEY = {v: k for k, v in CRITERION_TITLES.items()}

_CRIT_KEYWORDS = {
    "task_response": r"argument|example|support|position|opinion|prompt|task|idea|conclusion|introduction|"
                     r"analysis|develop|relevan|thesis|stance|both (views|sides)",
    "coherence_and_cohesion": r"transition|link|connect|cohesi|coheren|flow|paragraph|logical|organi[sz]|"
                              r"structure|sequenc|signpost|discourse|linking",
    "lexical_resource": r"vocabular|word choice|synonym|collocation|spelling|lexical|phrase|expression|repetit|"
                        r"terminology",
    "grammatical_range_and_accuracy": r"grammar|grammatical|tense|verb|article|preposition|punctuation|"
                                      r"sentence structure|subject|agreement|clause|complex sentence",
}


def parse_feedback_column(fb: str) -> dict:
    """Pull criterion-level improvement advice out of the free-text Feedback column.

    Deliberately ignores the 'Original:/Suggested:' rewrites: they hand the student a finished
    answer (and some contain invented facts), which is the opposite of guided feedback.
    """
    per_crit = {c: [] for c in CRITERIA}
    originals = {c: [] for c in CRITERIA}     # student sentences the reviewer pointed at
    overall = {}
    for m in _OVERALL_RE.finditer(fb or ""):
        overall[_TITLE_TO_KEY[m.group(1)]] = m.group(2).strip()
    # each block: "Improvement point: ...\nOriginal: "..."\nSuggested: ..."
    for blk in re.split(r"(?=^Improvement point:)", fb or "", flags=re.M):
        m = _IMPROVE_RE.match(blk)
        if not m:
            continue
        pt = m.group(1).strip()
        scores = {c: len(re.findall(p, pt, re.I)) for c, p in _CRIT_KEYWORDS.items()}
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            per_crit[best].append(pt)
            o = re.search(r"^Original:\s*[\"\u201c](.+?)[\"\u201d]\s*$", blk, re.M)
            if o:
                originals[best].append(o.group(1))
    return {"overall_line": overall, "points": per_crit, "originals": originals}


# ----------------------------------------------------------------------------- hint bank
# (regex on the weakness sentence) -> hint. {q} = a quotation from the essay related to the weakness.
_HINTS = {
    "task_response": [
        (r"conclusion", 'Reread your conclusion: does it restate your position in new words and sum up '
                        'the main reasons, without adding a new idea?'),
        (r"position|opinion|stance|thesis", 'Write your position in one clear sentence in the introduction, then check '
                                            'that each body paragraph supports it.'),
        (r"example|specific|support|elaborat|extend|develop|depth|detail|analysis|explain",
         'Pick one main idea{qq} and extend it: say WHY it is true, then add one concrete example. '
         'Aim for two or three connected sentences per idea.'),
        (r"both|all parts|fully address|partly|partially|misinterpret|irrelevant|off-topic|not address",
         'Underline each part of the question and note which paragraph answers it; any part without '
         'its own developed paragraph needs more attention.'),
        (r"250|short|length|word", 'Plan for at least 250 words: add one fully extended supporting idea rather than '
                                   'repeating points you have already made.'),
    ],
    "coherence_and_cohesion": [
        (r"paragraph", 'Check that each paragraph has one main idea, introduced by a clear topic sentence, and '
                       'that the order of paragraphs follows your plan.'),
        (r"repetit|reference|pronoun|substitut", 'Look for repeated nouns{qq}: replace some with pronouns '
                                                 '(it, this, they) or a suitable synonym so the reference stays clear.'),
        (r"mechanical|overuse|over-use|formulaic", 'Do not rely on the same linking words every time; '
                                                   'let the logic of the sentence decide the connector.'),
        (r"transition|link|connect|cohesi|flow|abrupt|jump|smooth|relationship|logical",
         'Take two neighbouring sentences{qq} and ask: what is the relationship (cause, contrast, example, result)? '
         'Make that relationship explicit with a suitable linker or reference word.'),
    ],
    "lexical_resource": [
        (r"spelling|misspell|word form|word formation", 'Check the spelling and word form of each key noun/verb{qq}: '
                                                         'is it the right part of speech (e.g. noun vs adjective)?'),
        (r"collocation|unnatural|awkward|inappropriate|imprecise|word choice|inaccurate|incorrect|precise",
         'Look at the word choice{qq}: look the main word up in a collocation dictionary and see which words '
         'normally go with it.'),
        (r"repetit|repeated|synonym|variety|limited|range|basic|simple", 'Underline words you use more than twice and '
                                                                          'decide which ones need a synonym or a more precise term.'),
        (r"topic|specific|sophisticat|less common|idiom", 'List three topic-specific words you could use for this '
                                                          'subject, and try each one in a sentence of your essay.'),
    ],
    "grammatical_range_and_accuracy": [
        (r"tense|verb form|subject-verb|agreement|verb", 'Underline the verb{qq}: check the tense, the verb form and '
                                                         'agreement with its subject.'),
        (r"article|preposition|determiner|plural|countable", 'Check each noun phrase{qq} for a missing or wrong article, '
                                                             'plural form or preposition.'),
        (r"punctuation|comma|run-on|fragment|splice|capital", 'Read the sentence{qq} aloud: where does one complete '
                                                              'idea end? Check commas, full stops and fragments.'),
        (r"complex|variety|range|simple|limited|structure", 'Try joining two short ideas with a subordinate clause '
                                                            '(although, which, when, because) and then re-check the verb forms.'),
        (r"error|inaccura|mistake|frequent|awkward", 'Proofread once for ONE error type at a time (verbs first, then articles, '
                                                     'then prepositions), starting with the sentence{qq}.'),
    ],
}
_GENERIC_HINT = {
    "task_response": 'Before writing, note your position and two supported reasons; after writing, check each paragraph '
                     'against that plan.',
    "coherence_and_cohesion": 'Read the essay aloud and mark every place where the reader has to guess how two ideas relate.',
    "lexical_resource": 'Read the essay once only for vocabulary and mark each word that feels vague, repeated or unusual.',
    "grammatical_range_and_accuracy": 'Read each sentence separately and check it has a subject, a correct verb and clear punctuation.',
}


def _quote_in(sentence: str) -> str | None:
    m = _QUOTE_RE.search(sentence)
    return m.group(1) if m else None


def hint_for(criterion: str, weakness: str, weak_quotes: list[str]) -> str | None:
    """A guiding hint for one weakness, anchored on a quotation that was cited as a PROBLEM."""
    q = _quote_in(weakness) or (weak_quotes[0] if weak_quotes else None)
    qq = f' (e.g. "{q}")' if q else ""
    for pat, tmpl in _HINTS[criterion]:
        if re.search(pat, weakness, re.I):
            return tmpl.format(qq=qq)
    return None


def build_suggestions(criterion: str, weaknesses: list[str], weak_quotes: list[str], overall_line: str | None,
                      points: list[str], underlength: bool, max_n: int = 3) -> list[str]:
    out: list[str] = []
    if underlength and criterion == "task_response":
        out.append("Plan for at least 250 words: add one fully extended supporting idea rather than repeating "
                   "points you have already made.")
    for w in weaknesses:
        h = hint_for(criterion, w, weak_quotes)
        if h and h not in out:
            out.append(h)
        if len(out) >= max_n - 1:
            break
    for extra in ([overall_line] if overall_line else []) + points[:1]:
        extra = (extra or "").strip()
        if extra and extra not in out and len(out) < max_n:
            if extra[-1] not in ".!?":
                extra += "."
            out.append(extra)
    return out[:max_n] or [_GENERIC_HINT[criterion]]


# ----------------------------------------------------------------------------- priorities / summary
def build_priorities(bands: dict, weaknesses: dict, evidence: dict, wc: int, max_n: int = 6) -> list[dict]:
    pri = []
    order = sorted(CRITERIA, key=lambda c: (bands[c], CRITERIA.index(c)))
    if wc < MIN_WORDS:
        pri.append({"criterion": "task_response",
                    "issue": f"The essay has only {wc} words, below the required 250, so ideas cannot be fully developed.",
                    "evidence": ""})
    def ev_of(w, c):
        q = _quote_in(w)
        if q:
            return q
        return ""
    # first pass: the main weakness of every criterion, weakest criterion first
    for c in order:
        if weaknesses[c]:
            w = weaknesses[c][0]
            pri.append({"criterion": c, "issue": re.sub(r'\s+', ' ', w), "evidence": ev_of(w, c) or (evidence[c][0] if evidence[c] else "")})
    # second pass: extra weaknesses, weakest criteria first
    for c in order:
        for w in weaknesses[c][1:2]:
            if len(pri) < max_n:
                pri.append({"criterion": c, "issue": re.sub(r'\s+', ' ', w),
                            "evidence": ev_of(w, c) or (evidence[c][1] if len(evidence[c]) > 1 else "")})
    pri = pri[:max_n]
    for i, p in enumerate(pri, 1):
        p["priority"] = i
    return [{"priority": p["priority"], "criterion": p["criterion"], "issue": p["issue"],
             "evidence": p["evidence"]} for p in pri]


def _band_phrase(b: float) -> str:
    if b >= 7:
        return "a strong response with good control and only occasional weaknesses"
    if b >= 6:
        return "a competent response whose message is clear but which has noticeable weaknesses"
    if b >= 5:
        return "a partly effective response with clear limitations that sometimes affect communication"
    return "a limited response with serious problems that frequently affect communication"


def _first_sentence(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) < 230 else s[:227].rsplit(" ", 1)[0] + "..."


def build_overall_feedback(overall: float, bands: dict, strengths: dict, weaknesses: dict, priorities: list,
                           wc: int, variant: int) -> str:
    best = max(CRITERIA, key=lambda c: (bands[c], -CRITERIA.index(c)))
    worst = min(CRITERIA, key=lambda c: (bands[c], CRITERIA.index(c)))
    parts = [f"This essay is estimated at Band {fmt_band(overall)} overall: {_band_phrase(overall)}."]
    if bands[best] == bands[worst]:
        parts.append(f"All four criteria are at a similar level (Band {fmt_band(bands[best])}).")
    else:
        sb = strengths[best][0] if strengths[best] else ""
        parts.append(f"The strongest criterion is {CRITERION_TITLES[best]} (Band {fmt_band(bands[best])})"
                     + (f": {_first_sentence(sb)}" if sb else "."))
        wb = weaknesses[worst][0] if weaknesses[worst] else ""
        lead = ("The weakest criterion is", "The criterion that most limits the score is",
                "The main limit on the score is")[variant % 3]
        parts.append(f"{lead} {CRITERION_TITLES[worst]} (Band {fmt_band(bands[worst])})"
                     + (f": {_first_sentence(wb)}" if wb else "."))
    if wc < MIN_WORDS:
        parts.append(f"The essay has {wc} words, fewer than the 250 required, which also limits Task Response.")
    if priorities:
        p = priorities[0]
        parts.append(f"Start with priority 1 ({CRITERION_TITLES[p['criterion']]}) and work through the priority "
                     "areas in order.")
    return " ".join(parts)
