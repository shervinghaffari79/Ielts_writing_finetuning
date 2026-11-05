"""Turn the assessment JSON into learner-friendly text, including a step-by-step (one priority at a time) view.

This is only a presentation layer. The step-by-step teaching dialogue itself belongs to the later
mediation stage - see docs/PROMPT_ENGINEERING.md - but this shows how the JSON is meant to be consumed.
"""
from __future__ import annotations

from .schema import CRITERIA, CRITERION_TITLES, fmt_band


def render_report(a: dict, max_priorities: int | None = None) -> str:
    L = [f"# Estimated overall band: {fmt_band(a['overall_band'])}",
         f"Words: {a['word_count']}" + ("  (**below the 250-word minimum**)" if a["underlength_warning"] else ""), ""]
    for c in CRITERIA:
        d = a["criteria"][c]
        L += [f"## {CRITERION_TITLES[c]} - Band {fmt_band(d['band'])}"]
        for title, key in (("Strengths", "strengths"), ("Weaknesses", "weaknesses"),
                           ("Evidence from your essay", "evidence"), ("How to improve", "improvement_suggestions")):
            L.append(f"**{title}**")
            L += [f"- {x}" if key != "evidence" else f'- "{x}"' for x in d[key]]
        L.append("")
    L.append("## Priority areas")
    for p in a["priority_areas"][:max_priorities]:
        ev = f' - "{p["evidence"]}"' if p["evidence"] else ""
        L.append(f"{p['priority']}. [{CRITERION_TITLES[p['criterion']]}] {p['issue']}{ev}")
    L += ["", "## Overall feedback", a["overall_feedback"]]
    return "\n".join(L)


def render_step(a: dict, step: int) -> str:
    """Progressive disclosure: show ONE priority (with its criterion's hints) instead of the whole report."""
    pri = a["priority_areas"]
    if not 1 <= step <= len(pri):
        return "No more steps. Rewrite your essay using what you learned, then submit it again."
    p = pri[step - 1]
    c = a["criteria"][p["criterion"]]
    lines = [f"Step {step} of {len(pri)} - {CRITERION_TITLES[p['criterion']]} (Band {fmt_band(c['band'])})",
             f"Focus: {p['issue']}"]
    if p["evidence"]:
        lines.append(f'Look at this part of your essay: "{p["evidence"]}"')
    if c["improvement_suggestions"]:
        lines.append("Try this: " + c["improvement_suggestions"][0])
    lines.append("Revise that part yourself, then ask for the next step.")
    return "\n".join(lines)
