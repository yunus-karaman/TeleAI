from __future__ import annotations

import re
from typing import Any

from data.schemas import KBParagraph, SolutionStep


FORBIDDEN_RULES = {
    "pii_request": [
        r"\btckn\b",
        r"\btc kimlik\b",
        r"\biban\b",
        r"\bimei\b",
        r"\biccid\b",
        r"\babone(?:lik)? no\b",
        r"\bmusteri no\b",
        r"\bkimlik (?:fotografi|taramasi|belgesi)\b",
        r"\badres(?:in)? tam\b",
    ],
    "overpromise": [
        r"\bkesin iade\b",
        r"\bgaranti iade\b",
        r"\bbugun duzelir\b",
        r"\bkesin cozulur\b",
        r"\b\d+\s*tl\b",
    ],
    "fake_legal": [
        r"\bkanun no\b",
        r"\byasa no\b",
        r"\bmevzuat\b",
        r"\bmadde\s*\d+\b",
    ],
    "hallucinated_policy": [
        r"\bpolicy[-_ ]?\d+\b",
        r"\bprosedur no\b",
        r"\bregulasyon no\b",
    ],
    "operator_specific": [
        r"\bturkcell\b",
        r"\bvodafone\b",
        r"\bturknet\b",
        r"\bturk telekom\b",
        r"\bbimcell\b",
        r"\bpttcell\b",
    ],
    "unsafe_instruction": [
        r"\belektrik panosu\b",
        r"\bkabloyu kes\b",
        r"\bhack\b",
        r"\byasadisi\b",
        r"\bguvenlik devre disi\b",
    ],
}

COMPILED_RULES = {
    rule_name: [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
    for rule_name, patterns in FORBIDDEN_RULES.items()
}


def _check_text(item_id: str, text: str, target: str) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for rule_name, patterns in COMPILED_RULES.items():
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                violations.append(
                    {
                        "item_id": item_id,
                        "target": target,
                        "rule": rule_name,
                        "pattern": pattern.pattern,
                        "snippet": text[max(0, match.start() - 30) : match.end() + 30],
                    }
                )
    return violations


def lint_solution_steps(steps: list[SolutionStep]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for step in steps:
        texts = [
            ("title_tr", step.title_tr),
            ("success_check", step.success_check),
            ("stop_conditions", " ".join(step.stop_conditions)),
            ("instructions_tr", " ".join(step.instructions_tr)),
        ]
        for target, text in texts:
            violations.extend(_check_text(item_id=step.step_id, text=text, target=target))

    return {
        "total_items": len(steps),
        "violations_count": len(violations),
        "violations": violations,
    }


def lint_kb_paragraphs(kb_items: list[KBParagraph]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for item in kb_items:
        violations.extend(_check_text(item_id=item.paragraph_id, text=item.text_tr, target="text_tr"))
    return {
        "total_items": len(kb_items),
        "violations_count": len(violations),
        "violations": violations,
    }

