"""Prompt construction. The SAME messages are used for open-source and OpenAI fine-tuning."""
from __future__ import annotations

from .schema import ORDER_SPEC, canonical_json, empty_assessment

# Short prompt: used for fine-tuning. The rubric knowledge is learned into the weights, so we do
# not pay (in tokens or money) to repeat a long rubric 450 times per epoch.
SYSTEM_PROMPT_SHORT = (
    "You are an IELTS Writing Task 2 examiner. Assess the student's essay against the four official "
    "criteria: Task Response, Coherence and Cohesion, Lexical Resource, Grammatical Range and "
    "Accuracy. Use only whole or half bands (0-9). Support every criterion with short, exact "
    "quotations from the essay and never invent text. Suggestions must be hints that help the "
    "student improve the essay themselves, not rewritten sentences. Reply with one valid JSON "
    "object in the required format and nothing else."
)

# Long prompt: used for the zero-shot / few-shot BASELINE (an un-tuned model, or an OpenAI model
# you do not fine-tune). It spells out everything the fine-tuned model learns by itself.
_RUBRIC = """\
Official key assessment criteria (IELTS Writing Task 2):
- Task Response (TR): how fully the task is answered; whether the position is clear and consistent;
  whether main ideas are extended and supported with relevant evidence/examples; relevance;
  clear introduction and conclusion; at least 250 words (a shorter essay limits TR).
- Coherence and Cohesion (CC): logical organisation and progression; paragraphing; sequencing within
  and across paragraphs; flexible use of reference/substitution; appropriate discourse markers and
  cohesive devices (not mechanical overuse).
- Lexical Resource (LR): range of vocabulary and synonyms; topic-specific and precise word choice;
  collocation, idiomatic and sophisticated phrasing; density and effect of spelling and
  word-formation errors.
- Grammatical Range and Accuracy (GRA): range of structures (simple, compound, complex); accuracy of
  sentences; density and communicative effect of errors; punctuation."""

_RULES = """\
Rules:
1. Assess ALL four criteria. Give each a band and an estimated overall band; use only whole or half
   bands between 0 and 9. Scores, explanations and evidence must agree with each other.
2. strengths / weaknesses must be specific to THIS essay and to the criterion being assessed.
3. evidence = short exact quotations copied from the essay (max ~25 words each). Never invent or
   "correct" a quotation.
4. improvement_suggestions = clear, practical hints tied to the weaknesses you found. Guide the
   student (what to look for, what to try). Do NOT rewrite the student's sentences for them.
5. If the essay has fewer than 250 words, set underlength_warning to true and take it into account
   in Task Response.
6. priority_areas: at most 8 problems, most important first (priority 1, 2, ...), each with the
   criterion key, a short issue description and one exact quotation (or "" if none applies).
7. Write in English. Output ONE valid JSON object using exactly the structure below, with no text or
   Markdown outside the JSON."""


def system_prompt_full(order: str = ORDER_SPEC) -> str:
    return (SYSTEM_PROMPT_SHORT + "\n\n" + _RUBRIC + "\n\n" + _RULES + "\n\nRequired JSON structure:\n"
            + canonical_json(empty_assessment(), order))


def user_message(prompt: str, essay: str, word_count_hint: bool = False) -> str:
    """The model input: exactly the task prompt and the essay (optionally a computed word count)."""
    parts = ["IELTS Writing Task 2 prompt:\n" + prompt.strip(), "Student essay:\n" + essay.strip()]
    if word_count_hint:
        parts.append(f"(Word count computed by the system: {len(essay.split())})")
    return "\n\n".join(parts)


def build_messages(prompt: str, essay: str, target_json: str | None = None, *, system: str | None = None,
                   word_count_hint: bool = False) -> list[dict]:
    msgs = [{"role": "system", "content": system or SYSTEM_PROMPT_SHORT},
            {"role": "user", "content": user_message(prompt, essay, word_count_hint)}]
    if target_json is not None:
        msgs.append({"role": "assistant", "content": target_json})
    return msgs
