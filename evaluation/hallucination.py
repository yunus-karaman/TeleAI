from __future__ import annotations

import re
from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from evaluation.common import write_json, write_markdown


STEP_ID_PATTERN = re.compile(r"STEP\.[A-Z0-9_]+\.\d{3}")
EVIDENCE_ID_PATTERN = re.compile(r"KB\.[A-Z0-9_]+\.\d{3}#P\d+")
TOKEN_PATTERN = re.compile(r"[a-z0-9çğıöşü]{2,}", flags=re.IGNORECASE)
MISMATCH_SIMILARITY_MIN = 0.08


def _solution_step_lines(text: str) -> list[str]:
    lines = [line.strip() for line in (text or "").splitlines()]
    start = None
    end = None
    for idx, line in enumerate(lines):
        if line.startswith("3) Çözüm Adımları"):
            start = idx + 1
        if line.startswith("4) Beklenen Sonuç / Kontrol:"):
            end = idx
            break
    if start is None:
        return [line for line in lines if line.startswith("- [STEP:")]
    selected = lines[start:end] if end is not None else lines[start:]
    return [line for line in selected if line.startswith("-")]


def _tokenize(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall((text or "").lower()))


def _strip_citation_markup(line: str) -> str:
    cleaned = STEP_ID_PATTERN.sub(" ", line or "")
    cleaned = EVIDENCE_ID_PATTERN.sub(" ", cleaned)
    cleaned = cleaned.replace("[STEP:", " ").replace("[KANIT:", " ").replace("]", " ")
    return " ".join(cleaned.split())


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / float(len(left | right))


def _score_case(case: dict[str, Any]) -> dict[str, Any]:
    inference = case["inference"]
    response_text = inference["response_text"]
    validation = inference["validation"]
    pack = inference["evidence_pack"]

    allowed_steps = {item["step_id"] for item in pack["top_steps"]}
    allowed_evidence = {item["paragraph_id"] for item in pack["evidence"]}

    referenced_steps = STEP_ID_PATTERN.findall(response_text)
    referenced_evidence = EVIDENCE_ID_PATTERN.findall(response_text)
    invalid_steps = [step_id for step_id in referenced_steps if step_id not in allowed_steps]
    invalid_evidence = [evidence_id for evidence_id in referenced_evidence if evidence_id not in allowed_evidence]

    evidence_text_by_id = {item.get("paragraph_id"): item.get("text_tr", "") for item in pack["evidence"]}

    actionable_lines = _solution_step_lines(response_text)
    actionable_claims = [line for line in actionable_lines if "[STEP:" in line]
    actionable_missing_evidence = [line for line in actionable_claims if len(EVIDENCE_ID_PATTERN.findall(line)) == 0]
    cited_actionable_claims = [line for line in actionable_claims if len(EVIDENCE_ID_PATTERN.findall(line)) > 0]

    mismatch_lines: list[str] = []
    for line in cited_actionable_claims:
        cited_ids = [evidence_id for evidence_id in EVIDENCE_ID_PATTERN.findall(line) if evidence_id in evidence_text_by_id]
        if not cited_ids:
            mismatch_lines.append(line)
            continue
        line_tokens = _tokenize(_strip_citation_markup(line))
        best_sim = 0.0
        for evidence_id in cited_ids:
            sim = _jaccard(line_tokens, _tokenize(evidence_text_by_id[evidence_id]))
            if sim > best_sim:
                best_sim = sim
        if best_sim < MISMATCH_SIMILARITY_MIN:
            mismatch_lines.append(line)

    return {
        "complaint_id": case["complaint_id"],
        "split": case["split"],
        "total_actionable_claims": len(actionable_claims),
        "hallucinated_actionable_claims": len(actionable_missing_evidence),
        "total_cited_actionable_claims": len(cited_actionable_claims),
        "evidence_mismatch_claims": len(mismatch_lines),
        "total_step_refs": len(referenced_steps),
        "invalid_step_refs": len(invalid_steps),
        "total_evidence_refs": len(referenced_evidence),
        "invalid_evidence_refs": len(invalid_evidence),
        "template_compliant": bool(validation.get("template_compliant", False)),
        "invalid_steps_sample": sorted(set(invalid_steps))[:5],
        "invalid_evidence_sample": sorted(set(invalid_evidence))[:5],
        "evidence_mismatch_sample": mismatch_lines[:3],
    }


def evaluate_hallucination(
    *,
    inference_cases: list[dict[str, Any]],
    report_json_path: str,
    report_md_path: str,
) -> dict[str, Any]:
    scored = [_score_case(case) for case in inference_cases]

    totals = Counter()
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    template_flags = []
    for item in scored:
        totals["total_actionable_claims"] += item["total_actionable_claims"]
        totals["hallucinated_actionable_claims"] += item["hallucinated_actionable_claims"]
        totals["total_cited_actionable_claims"] += item["total_cited_actionable_claims"]
        totals["evidence_mismatch_claims"] += item["evidence_mismatch_claims"]
        totals["total_step_refs"] += item["total_step_refs"]
        totals["invalid_step_refs"] += item["invalid_step_refs"]
        totals["total_evidence_refs"] += item["total_evidence_refs"]
        totals["invalid_evidence_refs"] += item["invalid_evidence_refs"]
        totals["cases"] += 1
        template_flags.append(1.0 if item["template_compliant"] else 0.0)

        split = item["split"]
        by_split[split]["cases"] += 1
        by_split[split]["total_actionable_claims"] += item["total_actionable_claims"]
        by_split[split]["hallucinated_actionable_claims"] += item["hallucinated_actionable_claims"]
        by_split[split]["total_cited_actionable_claims"] += item["total_cited_actionable_claims"]
        by_split[split]["evidence_mismatch_claims"] += item["evidence_mismatch_claims"]
        by_split[split]["total_step_refs"] += item["total_step_refs"]
        by_split[split]["invalid_step_refs"] += item["invalid_step_refs"]
        by_split[split]["total_evidence_refs"] += item["total_evidence_refs"]
        by_split[split]["invalid_evidence_refs"] += item["invalid_evidence_refs"]

    def _rate(num: int, den: int) -> float:
        return round(num / float(den), 6) if den > 0 else 0.0

    split_rates: dict[str, dict[str, float]] = {}
    for split, values in by_split.items():
        split_rates[split] = {
            "hallucination_rate_actionable": _rate(
                values["hallucinated_actionable_claims"], values["total_actionable_claims"]
            ),
            "evidence_mismatch_rate": _rate(values["evidence_mismatch_claims"], values["total_cited_actionable_claims"]),
            "step_hallucination_rate": _rate(values["invalid_step_refs"], values["total_step_refs"]),
            "citation_hallucination_rate": _rate(values["invalid_evidence_refs"], values["total_evidence_refs"]),
        }

    report = {
        "counts": {
            "cases": totals["cases"],
            "total_actionable_claims": totals["total_actionable_claims"],
            "hallucinated_actionable_claims": totals["hallucinated_actionable_claims"],
            "total_cited_actionable_claims": totals["total_cited_actionable_claims"],
            "evidence_mismatch_claims": totals["evidence_mismatch_claims"],
            "total_step_refs": totals["total_step_refs"],
            "invalid_step_refs": totals["invalid_step_refs"],
            "total_evidence_refs": totals["total_evidence_refs"],
            "invalid_evidence_refs": totals["invalid_evidence_refs"],
        },
        "metrics": {
            "hallucination_rate_actionable": _rate(
                totals["hallucinated_actionable_claims"], totals["total_actionable_claims"]
            ),
            "evidence_mismatch_rate": _rate(totals["evidence_mismatch_claims"], totals["total_cited_actionable_claims"]),
            "step_hallucination_rate": _rate(totals["invalid_step_refs"], totals["total_step_refs"]),
            "citation_hallucination_rate": _rate(totals["invalid_evidence_refs"], totals["total_evidence_refs"]),
            "template_compliance_rate": round(float(mean(template_flags)) if template_flags else 0.0, 6),
        },
        "per_split": split_rates,
        "sample_violations": [
            item
            for item in scored
            if item["hallucinated_actionable_claims"] > 0
            or item["evidence_mismatch_claims"] > 0
            or item["invalid_step_refs"] > 0
            or item["invalid_evidence_refs"] > 0
        ][:25],
    }
    write_json(report_json_path, report)

    lines = [
        "# Hallucination Report",
        "",
        f"- cases: `{report['counts']['cases']}`",
        f"- hallucination_rate_actionable: `{report['metrics']['hallucination_rate_actionable']}`",
        f"- evidence_mismatch_rate: `{report['metrics']['evidence_mismatch_rate']}`",
        f"- step_hallucination_rate: `{report['metrics']['step_hallucination_rate']}`",
        f"- citation_hallucination_rate: `{report['metrics']['citation_hallucination_rate']}`",
        f"- template_compliance_rate: `{report['metrics']['template_compliance_rate']}`",
        "",
        "## Per Split",
    ]
    for split in sorted(split_rates.keys()):
        payload = split_rates[split]
        lines.append(
            f"- {split}: actionable={payload['hallucination_rate_actionable']} | "
            f"mismatch={payload['evidence_mismatch_rate']} | "
            f"step={payload['step_hallucination_rate']} | citation={payload['citation_hallucination_rate']}"
        )
    write_markdown(report_md_path, lines)
    return report
