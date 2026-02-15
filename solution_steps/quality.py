from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any

from data.schemas import KBParagraph, SolutionStep
from solution_steps.generator import StepKBLink


def _stable_hash(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_solution_quality(
    *,
    steps: list[SolutionStep],
    kb_items: list[KBParagraph],
    links: list[StepKBLink],
    target_categories: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    min_steps = int(config["min_steps_per_category"])
    max_steps = int(config["max_steps_per_category"])
    min_l1 = int(config["min_level_counts"]["L1"])
    min_l2 = int(config["min_level_counts"]["L2"])
    min_l3 = int(config["min_level_counts"]["L3"])

    errors: list[str] = []
    steps_by_category: dict[str, list[SolutionStep]] = defaultdict(list)
    for step in steps:
        steps_by_category[step.category_id].append(step)

    count_per_category: dict[str, int] = {}
    level_counts: Counter[str] = Counter()
    for category_id in target_categories:
        category_steps = sorted(steps_by_category.get(category_id, []), key=lambda item: item.step_id)
        count_per_category[category_id] = len(category_steps)
        if not (min_steps <= len(category_steps) <= max_steps):
            errors.append(
                f"Category {category_id} has {len(category_steps)} steps; expected between {min_steps} and {max_steps}."
            )

        local_titles = [step.title_tr for step in category_steps]
        if len(local_titles) != len(set(local_titles)):
            errors.append(f"Category {category_id} has duplicated step titles.")

        category_levels = Counter(step.level for step in category_steps)
        level_counts.update(category_levels)
        if category_levels.get("L1", 0) < min_l1:
            errors.append(f"Category {category_id} must have at least {min_l1} L1 steps.")
        if category_levels.get("L2", 0) < min_l2:
            errors.append(f"Category {category_id} must have at least {min_l2} L2 steps.")
        if category_levels.get("L3", 0) < min_l3:
            errors.append(f"Category {category_id} must have at least {min_l3} L3 steps.")

        for step in category_steps:
            if not (3 <= len(step.instructions_tr) <= 6):
                errors.append(f"{step.step_id} has invalid instructions length: {len(step.instructions_tr)}")
            for bullet in step.instructions_tr:
                if len(bullet.strip()) < 10:
                    errors.append(f"{step.step_id} contains too-short instruction bullet.")

    step_ids = [step.step_id for step in steps]
    if len(step_ids) != len(set(step_ids)):
        errors.append("Duplicate step_id detected.")

    paragraph_ids = [item.paragraph_id for item in kb_items]
    if len(paragraph_ids) != len(set(paragraph_ids)):
        errors.append("Duplicate paragraph_id detected.")

    evidence_ids = set(paragraph_ids)
    links_by_step = {link.step_id: link for link in links}
    missing_evidence_steps = 0
    for step in steps:
        link = links_by_step.get(step.step_id)
        if link is None or len(link.evidence_ids) == 0:
            missing_evidence_steps += 1
            errors.append(f"{step.step_id} has no linked evidence.")
            continue
        for evidence_id in link.evidence_ids:
            if evidence_id not in evidence_ids:
                errors.append(f"{step.step_id} references missing evidence_id {evidence_id}.")

    hashes = {
        "steps_hash": _stable_hash([step.model_dump(mode="json") for step in sorted(steps, key=lambda item: item.step_id)]),
        "kb_hash": _stable_hash(
            [item.model_dump(mode="json") for item in sorted(kb_items, key=lambda item: item.paragraph_id)]
        ),
        "links_hash": _stable_hash(
            [link.model_dump(mode="json") for link in sorted(links, key=lambda item: item.step_id)]
        ),
    }

    return {
        "count_per_category": count_per_category,
        "count_per_level": dict(level_counts),
        "missing_evidence_count": missing_evidence_steps,
        "errors": errors,
        "hashes": hashes,
    }

