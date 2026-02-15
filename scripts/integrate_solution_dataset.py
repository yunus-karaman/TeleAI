from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from scripts.config_loader import load_config
from scripts.runtime_gates import handle_gate_violation
from scripts.solution_dataset_integrity import run_solution_dataset_integrity


ALLOWED_ESCALATION_UNITS = {
    "BILLING_SUPPORT",
    "TECH_SUPPORT_HOME",
    "TECH_SUPPORT_MOBILE",
    "NETWORK_NOC",
    "STORE",
    "PORTING_TEAM",
    "DIGITAL_SUPPORT",
    "GENERAL_SUPPORT",
}

ESCALATION_MAP = {
    "CUSTOMER_CARE": "GENERAL_SUPPORT",
    "STORE_SUPPORT": "STORE",
    "BILLING_SUPPORT": "BILLING_SUPPORT",
    "TECH_SUPPORT_HOME": "TECH_SUPPORT_HOME",
    "TECH_SUPPORT_MOBILE": "TECH_SUPPORT_MOBILE",
    "NETWORK_NOC": "NETWORK_NOC",
    "PORTING_TEAM": "PORTING_TEAM",
    "DIGITAL_SUPPORT": "DIGITAL_SUPPORT",
    "STORE": "STORE",
    "GENERAL_SUPPORT": "GENERAL_SUPPORT",
}

CATEGORY_ESCALATION_DEFAULT = {
    "BILLING_PAYMENTS": "BILLING_SUPPORT",
    "PLANS_PACKAGES": "BILLING_SUPPORT",
    "MOBILE_DATA_SPEED": "TECH_SUPPORT_MOBILE",
    "MOBILE_VOICE_SMS": "TECH_SUPPORT_MOBILE",
    "HOME_INTERNET_FIBER_DSL": "TECH_SUPPORT_HOME",
    "OUTAGE_SERVICE_DOWN": "NETWORK_NOC",
    "COVERAGE_SIGNAL": "NETWORK_NOC",
    "ROAMING_INTERNATIONAL": "TECH_SUPPORT_MOBILE",
    "NUMBER_PORTING_MNP": "PORTING_TEAM",
    "SIM_LINE_ACCOUNT": "STORE",
    "INSTALLATION_INFRASTRUCTURE": "TECH_SUPPORT_HOME",
    "MODEM_DEVICE": "TECH_SUPPORT_HOME",
    "DIGITAL_APP_AUTH": "DIGITAL_SUPPORT",
    "CUSTOMER_SUPPORT_PROCESS": "GENERAL_SUPPORT",
    "CONTRACT_CANCELLATION": "BILLING_SUPPORT",
    "OTHER": "GENERAL_SUPPORT",
}

RISK_LEVELS = {"low", "medium", "high"}
LEVELS = {"L1", "L2", "L3"}
TOKEN_PATTERN = re.compile(r"[a-z0-9çğıöşü]{2,}", flags=re.IGNORECASE)


def _semver(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "1.0.0"
    parts = text.split(".")
    if len(parts) == 1:
        return f"{parts[0]}.0.0"
    if len(parts) == 2:
        return f"{parts[0]}.{parts[1]}.0"
    return ".".join(parts[:3])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        for row in rows:
            outfile.write(json.dumps(row, ensure_ascii=False))
            outfile.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _ensure_list(value: Any) -> list[str]:
    if isinstance(value, list):
        result = [str(item).strip() for item in value if str(item).strip()]
    elif value is None:
        result = []
    else:
        text = str(value).strip()
        result = [text] if text else []
    deduped: list[str] = []
    seen = set()
    for item in result:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def _keywords_for_category(category_id: str, title: str, description: str) -> list[str]:
    base = [token for token in _tokenize(category_id.replace("_", " ")) if len(token) >= 2]
    base.extend(token for token in _tokenize(title) if len(token) >= 3)
    base.extend(token for token in _tokenize(description) if len(token) >= 3)
    deduped: list[str] = []
    seen = set()
    for token in base:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped[:12] if deduped else ["telekom", "sorun"]


def _map_escalation(unit: Any, category_id: str) -> str:
    candidate = str(unit or "").strip().upper()
    mapped = ESCALATION_MAP.get(candidate, candidate)
    if mapped in ALLOWED_ESCALATION_UNITS:
        return mapped
    fallback = CATEGORY_ESCALATION_DEFAULT.get(category_id, "GENERAL_SUPPORT")
    return fallback if fallback in ALLOWED_ESCALATION_UNITS else "GENERAL_SUPPORT"


def _normalize_taxonomy(raw_taxonomy: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_categories = raw_taxonomy.get("categories", [])
    taxonomy_version = _semver(raw_taxonomy.get("taxonomy_version") or raw_taxonomy.get("version"))
    normalized_rows: list[dict[str, Any]] = []
    actions = {"empty_keyword_fills": 0, "empty_example_fills": 0, "empty_escalation_fills": 0}

    for raw in sorted(raw_categories, key=lambda item: str(item.get("category_id", ""))):
        category_id = str(raw.get("category_id", "")).strip().upper()
        if not category_id:
            continue

        title = str(raw.get("title_tr") or category_id.replace("_", " ").title()).strip()
        description = str(raw.get("description_tr") or f"{title} ile ilgili telekom sorunlari").strip()
        if len(description) < 10:
            description = f"{description} ile ilgili telekom sorunlari"

        keywords = _ensure_list(raw.get("keywords_tr"))
        if not keywords:
            keywords = _keywords_for_category(category_id, title, description)
            actions["empty_keyword_fills"] += 1

        examples = _ensure_list(raw.get("example_phrases_tr"))
        if not examples:
            examples = [f"{title} ile ilgili sorun yasiyorum."]
            actions["empty_example_fills"] += 1

        negative_keywords = _ensure_list(raw.get("negative_keywords_tr"))
        escalation_default_unit = _map_escalation(raw.get("escalation_default_unit"), category_id)
        if not str(raw.get("escalation_default_unit") or "").strip():
            actions["empty_escalation_fills"] += 1

        risk_level = str(raw.get("risk_level_default") or "low").strip().lower()
        if risk_level not in RISK_LEVELS:
            risk_level = "low"

        normalized_rows.append(
            {
                "category_id": category_id,
                "title_tr": title,
                "description_tr": description,
                "keywords_tr": keywords,
                "negative_keywords_tr": negative_keywords,
                "example_phrases_tr": examples,
                "escalation_default_unit": escalation_default_unit,
                "risk_level_default": risk_level,
                "version": taxonomy_version,
            }
        )

    existing_ids = {item["category_id"] for item in normalized_rows}
    if "OTHER" not in existing_ids:
        normalized_rows.append(
            {
                "category_id": "OTHER",
                "title_tr": "Genel Diger",
                "description_tr": "Diger telekom sikayetleri icin genel sinif.",
                "keywords_tr": ["diger", "genel", "telekom"],
                "negative_keywords_tr": [],
                "example_phrases_tr": ["Sorunun kategorisini netlestiremedim."],
                "escalation_default_unit": "GENERAL_SUPPORT",
                "risk_level_default": "low",
                "version": taxonomy_version,
            }
        )

    normalized_rows = sorted(normalized_rows, key=lambda item: item["category_id"])
    taxonomy = {
        "taxonomy_name": str(raw_taxonomy.get("taxonomy_name") or "telecom_solution_dataset_v1"),
        "taxonomy_version": taxonomy_version,
        "language": "tr",
        "categories": normalized_rows,
    }
    actions["category_count"] = len(normalized_rows)
    return taxonomy, actions


def _normalize_steps(
    raw_steps: list[dict[str, Any]],
    valid_categories: set[str],
    taxonomy_version: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids = set()
    dropped_duplicate_ids = 0
    normalized_stop_conditions = 0
    mapped_escalations = 0
    dropped_unknown_category = 0

    for raw in sorted(raw_steps, key=lambda item: str(item.get("step_id", ""))):
        step_id = str(raw.get("step_id", "")).strip().upper()
        if not step_id:
            continue
        if step_id in seen_ids:
            dropped_duplicate_ids += 1
            continue
        seen_ids.add(step_id)

        category_id = str(raw.get("category_id", "")).strip().upper()
        if category_id not in valid_categories:
            dropped_unknown_category += 1
            continue

        level = str(raw.get("level") or "L1").strip().upper()
        if level not in LEVELS:
            level = "L1"

        title = str(raw.get("title_tr") or f"{category_id} adimi").strip()
        if len(title) < 5:
            title = f"{title} adimi"

        instructions = _ensure_list(raw.get("instructions_tr"))
        if not instructions:
            instructions = [
                "Sorunun ortaya ciktigi kosullari kisaca not edin.",
                "Ayni durumu kontrollu sekilde bir kez daha test edin.",
                "Gorulen uyari veya hata metnini kaydedin.",
            ]
        while len(instructions) < 3:
            instructions.append("Test sonucunu not edip bir sonraki adima gecin.")
        instructions = instructions[:6]
        normalized_instructions: list[str] = []
        for item in instructions:
            text = item.strip()
            if len(text) < 10:
                text = f"{text} Ayrintiyi kaydedin."
            normalized_instructions.append(text)

        required_inputs = _ensure_list(raw.get("required_inputs"))
        success_check = str(raw.get("success_check") or "Sorun durumu olculebilir sekilde netlesir.").strip()
        if len(success_check) < 8:
            success_check = f"{success_check} Sonucu dogrulayin."

        stop_conditions_value = raw.get("stop_conditions")
        stop_conditions = _ensure_list(stop_conditions_value)
        if not stop_conditions:
            stop_conditions = ["Sorun devam ederse ilgili destek birimine eskale edin."]
        if not isinstance(stop_conditions_value, list):
            normalized_stop_conditions += 1
        stop_conditions = stop_conditions[:4]

        raw_escalation = str(raw.get("escalation_unit") or "").strip().upper()
        escalation_unit = _map_escalation(raw_escalation, category_id)
        if escalation_unit != raw_escalation:
            mapped_escalations += 1

        risk_level = str(raw.get("risk_level") or ("medium" if level in {"L2", "L3"} else "low")).strip().lower()
        if risk_level not in RISK_LEVELS:
            risk_level = "medium" if level in {"L2", "L3"} else "low"

        tags = _ensure_list(raw.get("tags"))
        version = _semver(raw.get("version") or taxonomy_version)

        rows.append(
            {
                "schema_name": "SolutionStep",
                "schema_version": "1.0.0",
                "schema_revision": 1,
                "step_id": step_id,
                "category_id": category_id,
                "level": level,
                "title_tr": title,
                "instructions_tr": normalized_instructions,
                "required_inputs": required_inputs,
                "success_check": success_check,
                "stop_conditions": stop_conditions,
                "escalation_unit": escalation_unit,
                "risk_level": risk_level,
                "tags": tags,
                "version": version,
            }
        )

    actions = {
        "dropped_duplicate_step_ids": dropped_duplicate_ids,
        "normalized_stop_conditions": normalized_stop_conditions,
        "mapped_escalation_units": mapped_escalations,
        "dropped_steps_unknown_category": dropped_unknown_category,
    }
    return rows, actions


def _normalize_kb(
    raw_kb: list[dict[str, Any]],
    step_ids: set[str],
    taxonomy_version: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[str]]]:
    rows: list[dict[str, Any]] = []
    seen_paragraph_ids = set()
    dropped_duplicate_ids = 0
    truncated_text_count = 0
    removed_unknown_step_refs = 0
    kb_by_step: dict[str, list[str]] = defaultdict(list)

    for raw in sorted(raw_kb, key=lambda item: str(item.get("paragraph_id", ""))):
        paragraph_id = str(raw.get("paragraph_id", "")).strip().upper()
        if not paragraph_id:
            continue
        if paragraph_id in seen_paragraph_ids:
            dropped_duplicate_ids += 1
            continue
        seen_paragraph_ids.add(paragraph_id)

        doc_id = str(raw.get("doc_id") or paragraph_id.split("#", maxsplit=1)[0]).strip().upper()
        text = " ".join(str(raw.get("text_tr") or "").strip().split())
        if len(text) < 20:
            text = f"{text} Bu kanit metni adimin guvenli uygulanmasini aciklar."
        if len(text) > 600:
            text = text[:600].rstrip()
            truncated_text_count += 1

        applies_to_step_ids = [sid for sid in _ensure_list(raw.get("applies_to_step_ids")) if sid in step_ids]
        removed_unknown_step_refs += max(0, len(_ensure_list(raw.get("applies_to_step_ids"))) - len(applies_to_step_ids))
        source_type = "internal_best_practice"
        confidence_raw = raw.get("confidence", 0.75)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.75
        confidence = max(0.6, min(0.95, confidence))

        version = _semver(raw.get("version") or taxonomy_version)
        fallback_step = sorted(step_ids)[0] if step_ids else "STEP.OTHER.001"
        row = {
            "schema_name": "KBParagraph",
            "schema_version": "1.0.0",
            "schema_revision": 1,
            "doc_id": doc_id,
            "paragraph_id": paragraph_id,
            "text_tr": text,
            "applies_to_step_ids": applies_to_step_ids if applies_to_step_ids else [fallback_step],
            "source_type": source_type,
            "confidence": round(confidence, 6),
            "version": version,
        }
        rows.append(row)
        for step_id in row["applies_to_step_ids"]:
            kb_by_step[step_id].append(paragraph_id)

    for key in kb_by_step:
        kb_by_step[key] = sorted(set(kb_by_step[key]))

    actions = {
        "dropped_duplicate_kb_ids": dropped_duplicate_ids,
        "truncated_kb_text_count": truncated_text_count,
        "removed_unknown_step_refs_from_kb": removed_unknown_step_refs,
    }
    return rows, actions, kb_by_step


def _normalize_links(
    raw_links: list[dict[str, Any]],
    step_ids: set[str],
    kb_ids: set[str],
    kb_by_step: dict[str, list[str]],
    taxonomy_version: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    links_by_step: dict[str, dict[str, Any]] = {}
    dropped_duplicate_step_links = 0
    removed_unknown_step_links = 0
    removed_unknown_evidence_links = 0
    filled_missing_evidence = 0

    for raw in sorted(raw_links, key=lambda item: str(item.get("step_id", ""))):
        step_id = str(raw.get("step_id", "")).strip().upper()
        if step_id not in step_ids:
            removed_unknown_step_links += 1
            continue

        evidence_ids = [eid for eid in _ensure_list(raw.get("evidence_ids")) if eid in kb_ids]
        removed_unknown_evidence_links += max(0, len(_ensure_list(raw.get("evidence_ids"))) - len(evidence_ids))
        if not evidence_ids:
            evidence_ids = kb_by_step.get(step_id, [])[:2]
            if evidence_ids:
                filled_missing_evidence += 1

        rationale = str(raw.get("rationale") or "Adim ve kanit birbiriyle tutarlidir.").strip()
        if len(rationale) < 8:
            rationale = f"{rationale} Kanit ile desteklenir."
        version = _semver(raw.get("version") or taxonomy_version)

        if step_id in links_by_step:
            merged = sorted(set(links_by_step[step_id]["evidence_ids"]) | set(evidence_ids))
            links_by_step[step_id]["evidence_ids"] = merged
            dropped_duplicate_step_links += 1
            continue

        links_by_step[step_id] = {
            "step_id": step_id,
            "evidence_ids": sorted(set(evidence_ids)),
            "rationale": rationale,
            "version": version,
        }

    for step_id in sorted(step_ids):
        if step_id in links_by_step and links_by_step[step_id]["evidence_ids"]:
            continue
        fallback = kb_by_step.get(step_id, [])[:2]
        if not fallback:
            continue
        links_by_step[step_id] = {
            "step_id": step_id,
            "evidence_ids": fallback,
            "rationale": "Adim icin otomatik kanit eslestirmesi uygulandi.",
            "version": taxonomy_version,
        }
        filled_missing_evidence += 1

    rows = sorted(links_by_step.values(), key=lambda item: item["step_id"])
    actions = {
        "dropped_duplicate_step_links": dropped_duplicate_step_links,
        "removed_unknown_step_links": removed_unknown_step_links,
        "removed_unknown_evidence_links": removed_unknown_evidence_links,
        "filled_missing_evidence_links": filled_missing_evidence,
    }
    return rows, actions


def _integration_report_md(report: dict[str, Any]) -> list[str]:
    lines = [
        "# Solution Dataset Integration Report",
        "",
        f"- mode: `{report['mode']}`",
        f"- source_zip: `{report['source_zip']}`",
        f"- extracted_dir: `{report['extracted_dir']}`",
        "",
        "## Counts",
    ]
    for key, value in report["counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Normalization Actions")
    for key, value in report["actions"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append(f"- integrity_pass: `{report['integrity_pass']}`")
    if report.get("integrity_report_path"):
        lines.append(f"- integrity_report_path: `{report['integrity_report_path']}`")
    return lines


def integrate_solution_dataset(*, config: dict[str, Any], mode: str, zip_path: Path) -> dict[str, Any]:
    required_files = {"taxonomy.yaml", "solution_steps.jsonl", "kb.jsonl", "step_kb_links.jsonl", "README.md"}
    extraction_dir = Path(config["paths"]["artifacts_dir"]) / "tmp" / "solution_dataset_integration"
    extraction_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extraction_dir)

    discovered = {path.name for path in extraction_dir.rglob("*") if path.is_file()}
    missing = sorted(required_files - discovered)
    if missing:
        handle_gate_violation(
            config=config,
            mode=mode,
            stage="solution_dataset_integration",
            gate_key="schema_violation",
            reason_code="SOLUTION_DATASET_ARCHIVE_INVALID",
            message=f"Solution dataset zip missing required files: {', '.join(missing)}",
            details={"zip_path": str(zip_path)},
            logger=None,
        )
        raise RuntimeError("Solution dataset archive is missing required files.")

    taxonomy_src = next(extraction_dir.rglob("taxonomy.yaml"))
    steps_src = next(extraction_dir.rglob("solution_steps.jsonl"))
    kb_src = next(extraction_dir.rglob("kb.jsonl"))
    links_src = next(extraction_dir.rglob("step_kb_links.jsonl"))
    readme_src = next(extraction_dir.rglob("README.md"))

    taxonomy_raw = yaml.safe_load(taxonomy_src.read_text(encoding="utf-8"))
    step_rows_raw = _read_jsonl(steps_src)
    kb_rows_raw = _read_jsonl(kb_src)
    link_rows_raw = _read_jsonl(links_src)

    taxonomy_canonical, taxonomy_actions = _normalize_taxonomy(taxonomy_raw)
    category_ids = {item["category_id"] for item in taxonomy_canonical["categories"]}
    taxonomy_version = taxonomy_canonical["taxonomy_version"]

    steps_canonical, step_actions = _normalize_steps(
        raw_steps=step_rows_raw,
        valid_categories=category_ids,
        taxonomy_version=taxonomy_version,
    )
    step_ids = {row["step_id"] for row in steps_canonical}

    kb_canonical, kb_actions, kb_by_step = _normalize_kb(
        raw_kb=kb_rows_raw,
        step_ids=step_ids,
        taxonomy_version=taxonomy_version,
    )
    kb_ids = {row["paragraph_id"] for row in kb_canonical}

    links_canonical, link_actions = _normalize_links(
        raw_links=link_rows_raw,
        step_ids=step_ids,
        kb_ids=kb_ids,
        kb_by_step=kb_by_step,
        taxonomy_version=taxonomy_version,
    )

    taxonomy_target = Path(config["taxonomy"]["taxonomy_file"])
    steps_target = Path(config["paths"]["solution_steps_jsonl"])
    kb_target = Path(config["paths"]["kb_jsonl"])
    links_target = Path(config["paths"]["step_kb_links_jsonl"])
    readme_target = Path("docs/solution_dataset_README.md")

    taxonomy_target.parent.mkdir(parents=True, exist_ok=True)
    taxonomy_target.write_text(
        yaml.safe_dump(taxonomy_canonical, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    _write_jsonl(steps_target, steps_canonical)
    _write_jsonl(kb_target, kb_canonical)
    _write_jsonl(links_target, links_canonical)
    readme_target.parent.mkdir(parents=True, exist_ok=True)
    source_readme = readme_src.read_text(encoding="utf-8").rstrip()
    canonical_note = (
        "\n\n## Canonicalization Note\n"
        "- Integration normalizes records to project schema contracts.\n"
        "- `solution_steps.jsonl` and `kb.jsonl` include `schema_name`, `schema_version`, `schema_revision`.\n"
        "- `stop_conditions` is stored as a list for each step.\n"
    )
    readme_target.write_text(source_readme + canonical_note, encoding="utf-8")

    integrity_report = run_solution_dataset_integrity(config=config, mode=mode, logger=None, stage="solution_dataset_integrity")
    integration_report = {
        "mode": mode,
        "source_zip": str(zip_path),
        "extracted_dir": str(extraction_dir),
        "counts": {
            "taxonomy_categories_raw": len(taxonomy_raw.get("categories", [])),
            "taxonomy_categories_canonical": len(taxonomy_canonical["categories"]),
            "steps_raw": len(step_rows_raw),
            "steps_canonical": len(steps_canonical),
            "kb_raw": len(kb_rows_raw),
            "kb_canonical": len(kb_canonical),
            "links_raw": len(link_rows_raw),
            "links_canonical": len(links_canonical),
        },
        "actions": {**taxonomy_actions, **step_actions, **kb_actions, **link_actions},
        "integrity_pass": bool(integrity_report["overall_pass"]),
        "integrity_report_path": str(Path(config["paths"].get("integrity_dir", "artifacts/integrity")) / "solution_dataset_integrity_report.json"),
        "outputs": {
            "taxonomy": str(taxonomy_target),
            "solution_steps": str(steps_target),
            "kb": str(kb_target),
            "step_kb_links": str(links_target),
            "readme": str(readme_target),
        },
    }

    integrity_dir = Path(config["paths"].get("integrity_dir", "artifacts/integrity"))
    integration_json = integrity_dir / "solution_dataset_integration_report.json"
    integration_md = integrity_dir / "solution_dataset_integration_report.md"
    _write_json(integration_json, integration_report)
    _write_md(integration_md, _integration_report_md(integration_report))
    return integration_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Integrate telecom solution dataset archive into canonical project schema.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["SMOKE", "FULL"], default="FULL")
    parser.add_argument("--zip", dest="zip_path", default="telecom_solution_dataset_v1.zip")
    args = parser.parse_args()

    config = load_config(config_path=args.config, mode=args.mode)
    zip_path = Path(args.zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Solution dataset zip not found: {zip_path}")

    report = integrate_solution_dataset(config=config, mode=args.mode, zip_path=zip_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
