from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml
from pydantic import ValidationError

from data.schemas import CleanComplaint, GraphEdge, GraphNode, KBParagraph, NormalizedComplaint, RawComplaint, SolutionStep
from preprocess.pii import detect_pii_tags, sanitize_for_artifact
from preprocess.text_cleaning import assess_multi_complaint
from solution_steps.generator import StepKBLink
from solution_steps.linting import lint_kb_paragraphs, lint_solution_steps
from taxonomy.schema import load_taxonomy_file


SEVERITY_ORDER = {"P0": 0, "P1": 1, "P2": 2}
PROJECT_ROOT = Path(".").resolve()
AUDIT_DIR = PROJECT_ROOT / "artifacts" / "audit"


TELECOM_KEYWORDS = {
    "internet",
    "mobil",
    "hat",
    "fatura",
    "paket",
    "tarife",
    "sms",
    "arama",
    "sinyal",
    "sebeke",
    "modem",
    "wifi",
    "wi-fi",
    "fiber",
    "adsl",
    "vdsl",
    "sim",
    "roaming",
    "numara",
    "tasima",
    "taahhut",
    "mnp",
    "dns",
    "apn",
    "volte",
    "kesinti",
    "hiz",
    "odeme",
    "cekim",
}


ACTION_LEXICON = [
    "apn",
    "volte",
    "dns",
    "modem",
    "fiber",
    "wifi",
    "wi-fi",
    "roaming",
    "sim",
    "mnp",
    "numara tasima",
    "kesinti",
    "paket",
    "tarife",
    "fatura",
    "odeme",
    "sinyal",
    "reset",
    "kanal",
]


LEGAL_PATTERN = re.compile(r"\b(kanun|mevzuat|madde\s*\d+|yonetmelik)\b", flags=re.IGNORECASE)
REFUND_GUARANTEE_PATTERN = re.compile(r"\b(kesin iade|garanti iade|kesin cozulur|bugun duzelir)\b", flags=re.IGNORECASE)
OPERATOR_PATTERN = re.compile(r"\b(turkcell|vodafone|turk telekom|bimcell|turknet|pttcell)\b", flags=re.IGNORECASE)
AUDIT_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
AUDIT_IBAN_PATTERN = re.compile(r"\bTR\d{2}(?:\s?\d{4}){5}\s?\d{2}\b", flags=re.IGNORECASE)
AUDIT_PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d\s().\-]{8,}\d)(?!\w)")
AUDIT_11DIGIT_PATTERN = re.compile(r"(?<!\d)\d{11}(?!\d)")
AUDIT_DEVICE_PATTERN = re.compile(r"\b(?:imei|iccid|imsi|device id|cihaz id)\b[:#\s-]*[a-z0-9]{6,22}|(?<!\d)\d{15,22}(?!\d)", flags=re.IGNORECASE)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as infile:
        for line_number, raw_line in enumerate(infile, start=1):
            line = raw_line.strip()
            if not line:
                yield line_number, None, "EMPTY_LINE"
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                yield line_number, None, f"JSON_ERROR:{error}"
                continue
            yield line_number, payload, None


def _normalize_text_hash(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _is_probably_telecom(text: str) -> bool:
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in TELECOM_KEYWORDS)


def _severity_counts(findings: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"P0": 0, "P1": 0, "P2": 0}
    for finding in findings:
        counts[finding["severity"]] += 1
    return counts


def _sorted_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(findings, key=lambda item: (SEVERITY_ORDER.get(item["severity"], 99), item["id"]))


def _new_finding(
    *,
    finding_id: str,
    severity: str,
    title: str,
    impact: str,
    metrics: dict[str, Any] | None = None,
    evidence: list[str] | None = None,
    masked_samples: list[str] | None = None,
    fix_plan: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": finding_id,
        "severity": severity,
        "title": title,
        "impact": impact,
        "metrics": metrics or {},
        "evidence": evidence or [],
        "masked_samples": masked_samples or [],
        "fix_plan": fix_plan or [],
    }


def _render_md(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# {report['report_name']}")
    lines.append("")
    lines.append(f"- generated_at_utc: `{report['generated_at_utc']}`")
    lines.append(f"- overall_status: `{report['overall_status']}`")
    sev = report["severity_counts"]
    lines.append(f"- severity_counts: `P0={sev.get('P0', 0)}, P1={sev.get('P1', 0)}, P2={sev.get('P2', 0)}`")
    lines.append("")
    lines.append("## Metrics")
    for key, value in sorted(report.get("metrics", {}).items(), key=lambda item: item[0]):
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Findings")
    if not report.get("findings"):
        lines.append("- No findings.")
        return "\n".join(lines)

    for finding in report["findings"]:
        lines.append(f"### [{finding['severity']}] {finding['title']}")
        lines.append(f"- id: `{finding['id']}`")
        lines.append(f"- impact: {finding['impact']}")
        if finding.get("metrics"):
            compact_metrics = ", ".join(f"{k}={v}" for k, v in sorted(finding["metrics"].items(), key=lambda item: item[0]))
            lines.append(f"- metrics: `{compact_metrics}`")
        if finding.get("evidence"):
            for item in finding["evidence"]:
                lines.append(f"- evidence: {item}")
        if finding.get("masked_samples"):
            for sample in finding["masked_samples"][:5]:
                lines.append(f"- masked_sample: `{sample}`")
        if finding.get("fix_plan"):
            for idx, step in enumerate(finding["fix_plan"], start=1):
                lines.append(f"{idx}. {step}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_report(report_name: str, metrics: dict[str, Any], findings: list[dict[str, Any]], scope: list[str]) -> dict[str, Any]:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    if not findings:
        findings = [
            _new_finding(
                finding_id=f"{report_name.upper()}-INFO",
                severity="P2",
                title="No blocking issues detected by current heuristics",
                impact="Residual risk remains; this does not replace adversarial/manual review.",
                metrics={},
                evidence=[],
                masked_samples=["[NO_SAMPLE]"],
                fix_plan=["Keep this audit in CI and extend validators as new failure modes are observed."],
            )
        ]
    findings_sorted = _sorted_findings(findings)
    sev_counts = _severity_counts(findings_sorted)
    overall_status = "FAIL" if (sev_counts["P0"] > 0 or sev_counts["P1"] > 0) else "PASS"
    report = {
        "report_name": report_name,
        "generated_at_utc": _now_iso(),
        "scope": scope,
        "metrics": metrics,
        "severity_counts": sev_counts,
        "overall_status": overall_status,
        "findings": findings_sorted,
    }
    base = AUDIT_DIR / report_name
    (base.with_suffix(".json")).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (base.with_suffix(".md")).write_text(_render_md(report), encoding="utf-8")
    return report


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _list_python_files() -> list[Path]:
    rows: list[Path] = []
    for path in PROJECT_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rows.append(path)
    return sorted(rows)


def _module_name_from_path(path: Path) -> str:
    return ".".join(path.relative_to(PROJECT_ROOT).with_suffix("").parts)

def audit_code_quality(config: dict[str, Any]) -> dict[str, Any]:
    python_files = _list_python_files()
    test_files = [path for path in python_files if path.parts[0] == "tests"]
    core_files = [path for path in python_files if path.parts[0] != "tests"]

    module_index = {_module_name_from_path(path) for path in core_files}
    internal_roots = {module.split(".")[0] for module in module_index if module}

    todo_hits: list[str] = []
    broad_except_no_reraise: list[str] = []
    broad_except_with_pass: list[str] = []
    syntax_errors: list[str] = []
    internal_import_errors: list[str] = []
    fallback_hits: list[str] = []
    nondeterministic_hits: list[str] = []

    def module_exists(module_name: str) -> bool:
        if module_name in module_index:
            return True
        prefix = module_name + "."
        return any(item.startswith(prefix) for item in module_index)

    for path in core_files:
        rel = str(path.relative_to(PROJECT_ROOT))
        text = _read_text(path)

        for pattern in [r"\bTODO\b", r"\bFIXME\b", r"\bHACK\b"]:
            for match in re.finditer(pattern, text):
                todo_hits.append(f"{rel}:{text[:match.start()].count(chr(10)) + 1}")

        for pattern in [r"\bmock\b", r"\bfallback\b", r"error_fallback", r"renderer_fallback"]:
            if re.search(pattern, text, flags=re.IGNORECASE):
                fallback_hits.append(rel)
                break

        for pattern in [r"uuid\.uuid4\(", r"datetime\.now\(", r"request_id\s*=\s*f\"INFER-"]:
            if re.search(pattern, text):
                nondeterministic_hits.append(rel)
                break

        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError as error:
            syntax_errors.append(f"{rel}: {error}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    broad = False
                    if handler.type is None:
                        broad = True
                    elif isinstance(handler.type, ast.Name) and handler.type.id in {"Exception", "BaseException"}:
                        broad = True
                    has_raise = any(
                        isinstance(item, ast.Raise)
                        for item in ast.walk(ast.Module(body=handler.body, type_ignores=[]))
                    )
                    has_pass = any(
                        isinstance(item, ast.Pass)
                        for item in ast.walk(ast.Module(body=handler.body, type_ignores=[]))
                    )
                    if broad and not has_raise:
                        broad_except_no_reraise.append(f"{rel}:{handler.lineno}")
                    if broad and has_pass:
                        broad_except_with_pass.append(f"{rel}:{handler.lineno}")

            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    root = name.split(".")[0]
                    if root in internal_roots and not module_exists(name):
                        internal_import_errors.append(f"{rel}:{node.lineno} -> import {name}")
            if isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue
                if not node.module:
                    continue
                root = node.module.split(".")[0]
                if root in internal_roots and not module_exists(node.module):
                    internal_import_errors.append(f"{rel}:{node.lineno} -> from {node.module} import ...")

    core_module_stems = []
    for path in core_files:
        stem = path.stem
        if stem in {"__init__", "interface"}:
            continue
        if path.parts[0] not in {
            "api",
            "data",
            "evaluation",
            "graph",
            "kb",
            "models",
            "preprocess",
            "retrieval",
            "scripts",
            "solution_steps",
            "taxonomy",
            "training",
        }:
            continue
        core_module_stems.append((stem, str(path.relative_to(PROJECT_ROOT))))

    test_blob = "\n".join(str(path.relative_to(PROJECT_ROOT)).lower() for path in test_files)
    missing_test_modules = [module_path for stem, module_path in core_module_stems if stem.lower() not in test_blob]

    trainer_cfg = config.get("training_llm", {}).get("trainer", {})
    fallback_to_mock = bool(trainer_cfg.get("fallback_to_mock_on_failure", False))

    findings: list[dict[str, Any]] = []
    if syntax_errors or internal_import_errors:
        findings.append(
            _new_finding(
                finding_id="CODE-001",
                severity="P0",
                title="Broken Syntax/Import Risks",
                impact="Core pipeline modules may fail at runtime before safety checks can execute.",
                metrics={
                    "syntax_errors": len(syntax_errors),
                    "internal_import_errors": len(internal_import_errors),
                },
                evidence=(syntax_errors + internal_import_errors)[:10],
                masked_samples=[sanitize_for_artifact(item, max_chars=240) for item in (syntax_errors + internal_import_errors)[:5]],
                fix_plan=[
                    "Make syntax/import validation a mandatory preflight gate in FULL mode.",
                    "Fail pipeline startup when any internal import resolution check fails.",
                ],
            )
        )

    if broad_except_with_pass:
        findings.append(
            _new_finding(
                finding_id="CODE-002",
                severity="P0",
                title="Silent Exception Swallowing",
                impact="Exceptions can be hidden, producing false PASS metrics and silent degradation.",
                metrics={"broad_except_with_pass_count": len(broad_except_with_pass)},
                evidence=broad_except_with_pass[:10],
                masked_samples=[sanitize_for_artifact(item, max_chars=220) for item in broad_except_with_pass[:5]],
                fix_plan=[
                    "Replace silent handlers with explicit structured error logs and typed re-raise in FULL mode.",
                    "Keep warning-only behavior only in SMOKE mode and emit skip manifests.",
                ],
            )
        )

    if broad_except_no_reraise:
        findings.append(
            _new_finding(
                finding_id="CODE-003",
                severity="P1",
                title="Broad Exception Catch Without Re-raise",
                impact="Failure modes are merged, making root-cause analysis and fail-fast behavior unreliable.",
                metrics={"broad_except_no_reraise_count": len(broad_except_no_reraise)},
                evidence=broad_except_no_reraise[:10],
                masked_samples=[sanitize_for_artifact(item, max_chars=220) for item in broad_except_no_reraise[:5]],
                fix_plan=[
                    "Use typed exceptions and re-raise unexpected errors in FULL mode.",
                    "Record machine-readable error_code values for observability.",
                ],
            )
        )

    if fallback_to_mock or fallback_hits:
        findings.append(
            _new_finding(
                finding_id="CODE-004",
                severity="P0",
                title="Fallback/Mock Paths Present In Critical Flows",
                impact="FULL mode can silently rely on fallback behavior, violating deterministic production guarantees.",
                metrics={
                    "files_with_fallback_keywords": len(set(fallback_hits)),
                    "fallback_to_mock_on_failure": fallback_to_mock,
                },
                evidence=sorted(set(fallback_hits))[:12],
                masked_samples=[sanitize_for_artifact(item, max_chars=200) for item in sorted(set(fallback_hits))[:5]],
                fix_plan=[
                    "Disable mock and renderer fallback when mode=FULL.",
                    "Emit explicit aborted_reason artifacts instead of degraded response substitution.",
                ],
            )
        )

    if missing_test_modules:
        findings.append(
            _new_finding(
                finding_id="CODE-005",
                severity="P2",
                title="Potential Test Coverage Gaps",
                impact="Uncovered modules can regress without being detected by CI.",
                metrics={
                    "untested_module_candidates": len(missing_test_modules),
                    "total_core_modules": len(core_module_stems),
                },
                evidence=missing_test_modules[:15],
                masked_samples=[sanitize_for_artifact(item, max_chars=180) for item in missing_test_modules[:5]],
                fix_plan=[
                    "Add unit tests for uncovered modules, prioritizing inference/evaluation safety paths.",
                    "Enforce minimum per-module coverage threshold for core directories.",
                ],
            )
        )

    if nondeterministic_hits:
        findings.append(
            _new_finding(
                finding_id="CODE-006",
                severity="P2",
                title="Nondeterministic Runtime Identifiers Present",
                impact="Run artifacts differ across executions, reducing strict reproducibility.",
                metrics={"files_with_nondeterministic_calls": len(set(nondeterministic_hits))},
                evidence=sorted(set(nondeterministic_hits))[:10],
                masked_samples=[sanitize_for_artifact(item, max_chars=180) for item in sorted(set(nondeterministic_hits))[:5]],
                fix_plan=[
                    "Move random IDs/timestamps behind deterministic mode toggles for reproducibility-critical jobs.",
                    "Separate telemetry-only nondeterministic fields from model outputs.",
                ],
            )
        )

    metrics = {
        "python_file_count": len(core_files),
        "test_file_count": len(test_files),
        "todo_fixme_hack_count": len(todo_hits),
        "syntax_error_count": len(syntax_errors),
        "internal_import_error_count": len(internal_import_errors),
        "broad_except_no_reraise_count": len(broad_except_no_reraise),
        "broad_except_with_pass_count": len(broad_except_with_pass),
        "files_with_fallback_keywords": len(set(fallback_hits)),
        "files_with_nondeterministic_calls": len(set(nondeterministic_hits)),
        "untested_module_candidates": len(missing_test_modules),
    }
    return _write_report(
        "code_quality_report",
        metrics=metrics,
        findings=findings,
        scope=["codebase", "config.yaml"],
    )

def _scan_jsonl_with_model(path: Path, model_cls: Any, text_field: str, brand_field: str | None = None) -> tuple[list[Any], dict[str, Any]]:
    valid_rows: list[Any] = []
    invalid_samples: list[str] = []
    parse_errors = 0
    schema_errors = 0
    pii_hits = Counter()
    audit_pii_hits = Counter()
    pii_samples: list[str] = []
    short_count = 0
    non_telecom_count = 0
    multi_count = 0

    for line_number, payload, error in _iter_jsonl(path):
        if error:
            parse_errors += 1
            if len(invalid_samples) < 5:
                invalid_samples.append(f"{path}:{line_number}:{error}")
            continue
        try:
            row = model_cls.model_validate(payload)
        except ValidationError as exc:
            schema_errors += 1
            if len(invalid_samples) < 5:
                invalid_samples.append(
                    sanitize_for_artifact(
                        f"{path}:{line_number}:{json.dumps(exc.errors(include_input=False, include_url=False), ensure_ascii=False)}",
                        max_chars=220,
                    )
                )
            continue

        valid_rows.append(row)
        text = getattr(row, text_field, "") or ""
        brand = getattr(row, brand_field, None) if brand_field else None
        tags = detect_pii_tags(text, ignore_mask_tokens=True)
        if tags:
            for tag in tags:
                pii_hits[tag] += 1
            if len(pii_samples) < 5:
                pii_samples.append(sanitize_for_artifact(text, max_chars=240))
        if AUDIT_EMAIL_PATTERN.search(text):
            audit_pii_hits["EMAIL"] += 1
        if AUDIT_IBAN_PATTERN.search(text):
            audit_pii_hits["IBAN"] += 1
        if AUDIT_PHONE_PATTERN.search(text):
            audit_pii_hits["PHONE"] += 1
        if AUDIT_11DIGIT_PATTERN.search(text):
            audit_pii_hits["N11"] += 1
        if AUDIT_DEVICE_PATTERN.search(text):
            audit_pii_hits["DEVICE_ID"] += 1
        if audit_pii_hits and len(pii_samples) < 5:
            if any(
                (
                    AUDIT_EMAIL_PATTERN.search(text),
                    AUDIT_IBAN_PATTERN.search(text),
                    AUDIT_PHONE_PATTERN.search(text),
                    AUDIT_11DIGIT_PATTERN.search(text),
                    AUDIT_DEVICE_PATTERN.search(text),
                )
            ):
                pii_samples.append(sanitize_for_artifact(text, max_chars=240))

        if len(text.strip()) < 80:
            short_count += 1

        if not _is_probably_telecom(text):
            non_telecom_count += 1

        if assess_multi_complaint(text=text, brand_name=brand).suspected:
            multi_count += 1

    total = parse_errors + schema_errors + len(valid_rows)
    metrics = {
        "path": str(path),
        "total_lines": total,
        "valid_rows": len(valid_rows),
        "parse_error_count": parse_errors,
        "schema_error_count": schema_errors,
        "pii_hits": dict(pii_hits),
        "audit_pii_hits": dict(audit_pii_hits),
        "short_count_lt_80": short_count,
        "non_telecom_candidate_count": non_telecom_count,
        "multi_complaint_suspected_count": multi_count,
    }
    return valid_rows, {"metrics": metrics, "invalid_samples": invalid_samples, "pii_samples": pii_samples}


def audit_data_quality() -> tuple[dict[str, Any], list[CleanComplaint], list[NormalizedComplaint]]:
    raw_path = PROJECT_ROOT / "sikayetler.jsonl"
    clean_path = PROJECT_ROOT / "artifacts" / "complaints_clean.jsonl"
    labeled_path = PROJECT_ROOT / "artifacts" / "complaints_labeled.jsonl"

    _, raw_meta = _scan_jsonl_with_model(raw_path, RawComplaint, text_field="complaint_text", brand_field="brand_name")
    clean_rows, clean_meta = _scan_jsonl_with_model(
        clean_path, CleanComplaint, text_field="complaint_text_clean", brand_field="brand_name"
    )
    labeled_rows, labeled_meta = _scan_jsonl_with_model(
        labeled_path, NormalizedComplaint, text_field="complaint_text_clean", brand_field="brand_name"
    )

    clean_by_id = {row.complaint_id: row for row in clean_rows}
    labeled_by_id = {row.complaint_id: row for row in labeled_rows}
    missing_in_labeled = sorted(set(clean_by_id) - set(labeled_by_id))
    missing_in_clean = sorted(set(labeled_by_id) - set(clean_by_id))

    source_hash_counts = Counter(row.source_hash_sha256 for row in clean_rows)
    duplicate_source_hash_groups = sum(1 for count in source_hash_counts.values() if count > 1)
    duplicate_source_hash_rows = sum(count - 1 for count in source_hash_counts.values() if count > 1)

    text_hash_counts = Counter(_normalize_text_hash(row.complaint_text_clean) for row in clean_rows)
    duplicate_text_groups = sum(1 for count in text_hash_counts.values() if count > 1)
    duplicate_text_rows = sum(count - 1 for count in text_hash_counts.values() if count > 1)

    findings: list[dict[str, Any]] = []

    for dataset_name, meta in [("raw", raw_meta), ("clean", clean_meta), ("labeled", labeled_meta)]:
        parse_err = meta["metrics"]["parse_error_count"]
        schema_err = meta["metrics"]["schema_error_count"]
        if parse_err or schema_err:
            findings.append(
                _new_finding(
                    finding_id=f"DATA-{dataset_name.upper()}-SCHEMA",
                    severity="P0",
                    title=f"{dataset_name} dataset schema/parse failures",
                    impact="Invalid records bypass quality controls and can break deterministic downstream behavior.",
                    metrics={"parse_error_count": parse_err, "schema_error_count": schema_err},
                    evidence=meta["invalid_samples"][:5],
                    masked_samples=meta["invalid_samples"][:5],
                    fix_plan=[
                        "Enforce strict schema gate before accepting records into artifacts.",
                        "Quarantine invalid records with explicit reason codes and block FULL mode.",
                    ],
                )
            )

    clean_pii_total = sum(clean_meta["metrics"]["pii_hits"].values())
    labeled_pii_total = sum(labeled_meta["metrics"]["pii_hits"].values())
    clean_pii_audit_total = sum(clean_meta["metrics"]["audit_pii_hits"].values())
    labeled_pii_audit_total = sum(labeled_meta["metrics"]["audit_pii_hits"].values())
    raw_pii_audit_total = sum(raw_meta["metrics"]["audit_pii_hits"].values())
    if clean_pii_total > 0 or labeled_pii_total > 0 or clean_pii_audit_total > 0 or labeled_pii_audit_total > 0:
        findings.append(
            _new_finding(
                finding_id="DATA-PII-LEAK",
                severity="P0",
                title="Residual Raw PII Detected In Processed Artifacts",
                impact="KVKK risk: sanitized artifacts still include detectable personal identifiers.",
                metrics={
                    "clean_pii_hits": clean_pii_total,
                    "labeled_pii_hits": labeled_pii_total,
                    "clean_pii_hits_audit_regex": clean_pii_audit_total,
                    "labeled_pii_hits_audit_regex": labeled_pii_audit_total,
                    "clean_pii_breakdown": clean_meta["metrics"]["pii_hits"],
                    "labeled_pii_breakdown": labeled_meta["metrics"]["pii_hits"],
                    "clean_pii_breakdown_audit": clean_meta["metrics"]["audit_pii_hits"],
                    "labeled_pii_breakdown_audit": labeled_meta["metrics"]["audit_pii_hits"],
                },
                evidence=["artifacts/complaints_clean.jsonl", "artifacts/complaints_labeled.jsonl"],
                masked_samples=(clean_meta["pii_samples"] + labeled_meta["pii_samples"])[:5],
                fix_plan=[
                    "Run strict second-pass masking and fail FULL mode when any raw PII tag remains.",
                    "Add regression tests specifically for phone/email/IBAN/TCKN/device/account patterns.",
                ],
            )
        )

    if raw_pii_audit_total > 0:
        findings.append(
            _new_finding(
                finding_id="DATA-RAW-PII",
                severity="P2",
                title="Raw Source Contains Extensive Unmasked PII",
                impact="Raw dataset handling requires strict access control and non-export safeguards.",
                metrics={
                    "raw_pii_hits_audit_regex": raw_pii_audit_total,
                    "raw_pii_breakdown_audit": raw_meta["metrics"]["audit_pii_hits"],
                },
                evidence=["sikayetler.jsonl"],
                masked_samples=raw_meta["pii_samples"][:5],
                fix_plan=[
                    "Restrict raw dataset access and enforce encrypted-at-rest controls.",
                    "Ensure only masked derivatives are used in downstream artifacts.",
                ],
            )
        )

    if duplicate_source_hash_rows > 0 or duplicate_text_rows > 0:
        findings.append(
            _new_finding(
                finding_id="DATA-DUPLICATES",
                severity="P1",
                title="Near/Exact Duplicates Present In Clean Artifact",
                impact="Duplicates can inflate evaluation metrics and bias retrieval behavior.",
                metrics={
                    "duplicate_source_hash_groups": duplicate_source_hash_groups,
                    "duplicate_source_hash_rows": duplicate_source_hash_rows,
                    "duplicate_text_groups": duplicate_text_groups,
                    "duplicate_text_rows": duplicate_text_rows,
                },
                evidence=["artifacts/complaints_clean.jsonl"],
                masked_samples=[sanitize_for_artifact(row.complaint_text_clean, max_chars=240) for row in clean_rows[:2]],
                fix_plan=[
                    "Apply deterministic duplicate pruning in FULL mode with canonical record retention.",
                    "Track duplicate clusters across all downstream splits.",
                ],
            )
        )

    clean_multi = clean_meta["metrics"]["multi_complaint_suspected_count"]
    if clean_rows and (clean_multi / len(clean_rows)) > 0.01:
        findings.append(
            _new_finding(
                finding_id="DATA-MULTI-COMPLAINT",
                severity="P1",
                title="Multi-Complaint Contamination Remains",
                impact="Single-label assignment becomes noisy when multiple complaint intents remain in one record.",
                metrics={
                    "multi_complaint_count": clean_multi,
                    "multi_complaint_rate": round(clean_multi / len(clean_rows), 6),
                },
                evidence=["artifacts/complaints_clean.jsonl"],
                masked_samples=[sanitize_for_artifact(row.complaint_text_clean, max_chars=220) for row in clean_rows[:3]],
                fix_plan=[
                    "Enforce split/truncate strategy with explicit quality flag for unresolved multi-intent entries.",
                    "Block FULL mode when contamination exceeds configured threshold.",
                ],
            )
        )

    clean_non_tel = clean_meta["metrics"]["non_telecom_candidate_count"]
    if clean_rows and (clean_non_tel / len(clean_rows)) > 0.01:
        findings.append(
            _new_finding(
                finding_id="DATA-NON-TELECOM",
                severity="P1",
                title="Non-Telecom Candidate Records Detected",
                impact="Out-of-domain complaints reduce category precision and retrieval relevance.",
                metrics={
                    "non_telecom_candidate_count": clean_non_tel,
                    "non_telecom_candidate_rate": round(clean_non_tel / len(clean_rows), 6),
                },
                evidence=["artifacts/complaints_clean.jsonl"],
                masked_samples=[sanitize_for_artifact(row.complaint_text_clean, max_chars=220) for row in clean_rows[:3]],
                fix_plan=[
                    "Add telecom-domain whitelist/blacklist filter before taxonomy assignment.",
                    "Route uncertain out-of-domain records to OTHER with strict review queue.",
                ],
            )
        )

    if missing_in_labeled or missing_in_clean:
        findings.append(
            _new_finding(
                finding_id="DATA-CONSISTENCY",
                severity="P1",
                title="Clean/Labeled Dataset ID Mismatch",
                impact="Inconsistent record lineage undermines reproducibility across stages.",
                metrics={
                    "missing_in_labeled": len(missing_in_labeled),
                    "missing_in_clean": len(missing_in_clean),
                },
                evidence=[
                    f"missing_in_labeled_sample={missing_in_labeled[:5]}",
                    f"missing_in_clean_sample={missing_in_clean[:5]}",
                ],
                masked_samples=[
                    sanitize_for_artifact(",".join(missing_in_labeled[:5]), max_chars=200),
                    sanitize_for_artifact(",".join(missing_in_clean[:5]), max_chars=200),
                ],
                fix_plan=[
                    "Add strict lineage check between clean and labeled artifacts before taxonomy stage completes.",
                    "Abort FULL mode on any record ID mismatch.",
                ],
            )
        )

    metrics = {
        "raw_total_lines": raw_meta["metrics"]["total_lines"],
        "raw_valid_rows": raw_meta["metrics"]["valid_rows"],
        "clean_total_lines": clean_meta["metrics"]["total_lines"],
        "clean_valid_rows": clean_meta["metrics"]["valid_rows"],
        "labeled_total_lines": labeled_meta["metrics"]["total_lines"],
        "labeled_valid_rows": labeled_meta["metrics"]["valid_rows"],
        "clean_pii_hits_total": clean_pii_total,
        "labeled_pii_hits_total": labeled_pii_total,
        "clean_pii_hits_audit_total": clean_pii_audit_total,
        "labeled_pii_hits_audit_total": labeled_pii_audit_total,
        "raw_pii_hits_audit_total": raw_pii_audit_total,
        "duplicate_source_hash_rows": duplicate_source_hash_rows,
        "duplicate_text_rows": duplicate_text_rows,
        "clean_multi_complaint_suspected_count": clean_multi,
        "clean_non_telecom_candidate_count": clean_non_tel,
        "id_mismatch_missing_in_labeled": len(missing_in_labeled),
        "id_mismatch_missing_in_clean": len(missing_in_clean),
    }
    report = _write_report(
        "data_quality_report",
        metrics=metrics,
        findings=findings,
        scope=["sikayetler.jsonl", "artifacts/complaints_clean.jsonl", "artifacts/complaints_labeled.jsonl"],
    )
    return report, clean_rows, labeled_rows

def audit_taxonomy(labeled_rows: list[NormalizedComplaint]) -> dict[str, Any]:
    taxonomy_path = PROJECT_ROOT / "taxonomy" / "taxonomy.yaml"
    taxonomy = load_taxonomy_file(str(taxonomy_path))
    categories = [category.category_id for category in taxonomy.categories]
    category_counts = Counter(row.normalized_category for row in labeled_rows)

    non_zero_counts = [count for count in category_counts.values() if count > 0]
    imbalance_ratio = round((max(non_zero_counts) / min(non_zero_counts)), 6) if non_zero_counts else 0.0
    other_rate = round((category_counts.get("OTHER", 0) / len(labeled_rows)), 6) if labeled_rows else 0.0

    confidence_values = [row.confidence_score for row in labeled_rows]
    low_conf_055 = sum(1 for value in confidence_values if value < 0.55)
    low_conf_065 = sum(1 for value in confidence_values if value < 0.65)
    conf_stats = {
        "mean": round(mean(confidence_values), 6) if confidence_values else 0.0,
        "median": round(median(confidence_values), 6) if confidence_values else 0.0,
        "min": round(min(confidence_values), 6) if confidence_values else 0.0,
        "max": round(max(confidence_values), 6) if confidence_values else 0.0,
        "rate_lt_055": round(low_conf_055 / len(confidence_values), 6) if confidence_values else 0.0,
        "rate_lt_065": round(low_conf_065 / len(confidence_values), 6) if confidence_values else 0.0,
    }

    taxonomy_report_path = PROJECT_ROOT / "artifacts" / "taxonomy_report.json"
    disagreement_rate = None
    if taxonomy_report_path.exists():
        try:
            report_payload = json.loads(taxonomy_report_path.read_text(encoding="utf-8"))
            disagreement_rate = report_payload.get("rule_embedding_disagreement_rate")
        except json.JSONDecodeError:
            disagreement_rate = None

    confusion_pairs_path = PROJECT_ROOT / "artifacts" / "error_analysis" / "confusion_pair_examples.json"
    top_confusions: list[tuple[str, int]] = []
    confusion_samples: list[str] = []
    if confusion_pairs_path.exists():
        try:
            confusion_payload = json.loads(confusion_pairs_path.read_text(encoding="utf-8"))
            if isinstance(confusion_payload, dict):
                top_confusions = sorted(
                    ((pair, len(items)) for pair, items in confusion_payload.items() if isinstance(items, list)),
                    key=lambda item: (-item[1], item[0]),
                )[:10]
                for pair, items in confusion_payload.items():
                    if not isinstance(items, list) or not items:
                        continue
                    sample = items[0]
                    snippet = sample.get("text_snippet", "")
                    confusion_samples.append(f"{pair}: {sanitize_for_artifact(snippet, max_chars=200)}")
                    if len(confusion_samples) >= 5:
                        break
        except json.JSONDecodeError:
            top_confusions = []

    findings: list[dict[str, Any]] = []
    if not (15 <= len(categories) <= 30):
        findings.append(
            _new_finding(
                finding_id="TAX-001",
                severity="P1",
                title="Taxonomy Category Count Outside Target",
                impact="Coverage breadth may be too coarse or too fragmented for stable operations.",
                metrics={"category_count": len(categories)},
                evidence=[str(taxonomy_path)],
                fix_plan=[
                    "Refine taxonomy to 15-30 stable telecom categories with explicit inclusion/exclusion definitions.",
                ],
            )
        )

    if imbalance_ratio > 20:
        findings.append(
            _new_finding(
                finding_id="TAX-002",
                severity="P1",
                title="Severe Category Imbalance",
                impact="Minor classes underfit and escalation policies become unstable.",
                metrics={"imbalance_ratio": imbalance_ratio, "min_class": min(non_zero_counts), "max_class": max(non_zero_counts)},
                evidence=[f"category_counts_top={category_counts.most_common(5)}"],
                fix_plan=[
                    "Use stratified rebalancing and targeted curation for low-support categories.",
                    "Generate GOLD review sets with per-class minimum quotas.",
                ],
            )
        )

    if other_rate > 0.08:
        findings.append(
            _new_finding(
                finding_id="TAX-003",
                severity="P1",
                title="OTHER Category Overuse",
                impact="High OTHER volume hides actionable telecom intents and weakens routing quality.",
                metrics={"other_rate": other_rate, "other_count": category_counts.get("OTHER", 0)},
                evidence=["artifacts/complaints_labeled.jsonl"],
                masked_samples=[sanitize_for_artifact(row.complaint_text_clean, max_chars=220) for row in labeled_rows[:3]],
                fix_plan=[
                    "Tighten OTHER fallback thresholds and add category-specific negative rules.",
                    "Route low-confidence OTHER records into human review loop.",
                ],
            )
        )

    if conf_stats["rate_lt_065"] > 0.15:
        findings.append(
            _new_finding(
                finding_id="TAX-004",
                severity="P1",
                title="Weak Confidence Distribution",
                impact="Low-confidence predictions increase wrong-step risk in downstream assistance.",
                metrics=conf_stats,
                evidence=["artifacts/complaints_labeled.jsonl"],
                masked_samples=[sanitize_for_artifact(row.assignment_reason, max_chars=220) for row in labeled_rows[:5]],
                fix_plan=[
                    "Calibrate confidence scores and enforce review margin stricter thresholds.",
                    "Blend rule + embedding outputs with validated calibration set.",
                ],
            )
        )

    if disagreement_rate is not None and disagreement_rate > 0.35:
        findings.append(
            _new_finding(
                finding_id="TAX-005",
                severity="P1",
                title="High Rule/Embedding Disagreement",
                impact="Hybrid assigner components conflict frequently, reducing label stability.",
                metrics={"rule_embedding_disagreement_rate": disagreement_rate},
                evidence=["artifacts/taxonomy_report.json"],
                fix_plan=[
                    "Review conflicting terms and retrain embedding baseline with domain-balanced examples.",
                    "Create confusion-pair remediation rules for top disagreements.",
                ],
            )
        )

    if top_confusions:
        findings.append(
            _new_finding(
                finding_id="TAX-006",
                severity="P2",
                title="Frequent Confusion Pairs Identified",
                impact="Known pairwise confusions require targeted disambiguation heuristics.",
                metrics={"top_confusion_pairs": top_confusions[:5]},
                evidence=[f"{pair}={count}" for pair, count in top_confusions[:8]],
                masked_samples=confusion_samples[:5],
                fix_plan=[
                    "Add pair-specific differentiator keywords and curated examples.",
                    "Track confusion-pair trend metrics as hard safety gate inputs.",
                ],
            )
        )

    metrics = {
        "category_count": len(categories),
        "record_count": len(labeled_rows),
        "imbalance_ratio": imbalance_ratio,
        "other_rate": other_rate,
        "confidence_mean": conf_stats["mean"],
        "confidence_rate_lt_055": conf_stats["rate_lt_055"],
        "confidence_rate_lt_065": conf_stats["rate_lt_065"],
        "rule_embedding_disagreement_rate": disagreement_rate,
    }
    return _write_report(
        "taxonomy_report",
        metrics=metrics,
        findings=findings,
        scope=["taxonomy/taxonomy.yaml", "artifacts/complaints_labeled.jsonl", "artifacts/error_analysis/confusion_pair_examples.json"],
    )

def _load_jsonl_models(path: Path, model_cls: Any) -> tuple[list[Any], list[str]]:
    rows: list[Any] = []
    errors: list[str] = []
    for line_number, payload, error in _iter_jsonl(path):
        if error:
            errors.append(f"{path}:{line_number}:{error}")
            continue
        try:
            rows.append(model_cls.model_validate(payload))
        except ValidationError as exc:
            errors.append(
                sanitize_for_artifact(
                    f"{path}:{line_number}:{json.dumps(exc.errors(include_input=False, include_url=False), ensure_ascii=False)}",
                    max_chars=220,
                )
            )
    return rows, errors


def audit_solution_steps() -> tuple[dict[str, Any], list[SolutionStep]]:
    steps_path = PROJECT_ROOT / "artifacts" / "solution_steps.jsonl"
    steps, step_errors = _load_jsonl_models(steps_path, SolutionStep)
    lint_report = lint_solution_steps(steps)

    steps_per_category = Counter(step.category_id for step in steps)
    level_counts = Counter(step.level for step in steps)

    duplicate_titles = Counter(step.title_tr.strip().lower() for step in steps)
    duplicate_title_count = sum(1 for value in duplicate_titles.values() if value > 1)

    instruction_templates = Counter(" || ".join(item.strip().lower() for item in step.instructions_tr) for step in steps)
    highly_reused_templates = {template: count for template, count in instruction_templates.items() if count > 2}

    action_coverage_missing = 0
    action_missing_samples: list[str] = []
    for step in steps:
        corpus = " ".join([step.title_tr] + step.instructions_tr).lower()
        has_action = any(token in corpus for token in ACTION_LEXICON)
        if not has_action:
            action_coverage_missing += 1
            if len(action_missing_samples) < 5:
                action_missing_samples.append(sanitize_for_artifact(corpus, max_chars=200))

    category_alignment_errors = []
    for step in steps:
        parts = step.step_id.split(".")
        if len(parts) < 3 or parts[1] != step.category_id:
            category_alignment_errors.append(step.step_id)

    low_depth_categories = sorted(category for category, count in steps_per_category.items() if count < 8)

    legal_hits = 0
    guarantee_hits = 0
    operator_hits = 0
    for step in steps:
        text = " ".join([step.title_tr, step.success_check, " ".join(step.stop_conditions), " ".join(step.instructions_tr)])
        if LEGAL_PATTERN.search(text):
            legal_hits += 1
        if REFUND_GUARANTEE_PATTERN.search(text):
            guarantee_hits += 1
        if OPERATOR_PATTERN.search(text):
            operator_hits += 1

    findings: list[dict[str, Any]] = []

    if step_errors:
        findings.append(
            _new_finding(
                finding_id="STEP-001",
                severity="P0",
                title="SolutionStep Schema Violations",
                impact="Invalid step records can break retrieval and response validation chain.",
                metrics={"schema_error_count": len(step_errors)},
                evidence=step_errors[:10],
                masked_samples=step_errors[:5],
                fix_plan=[
                    "Block stage completion when any SolutionStep schema validation fails.",
                    "Add pre-commit schema validation for generated step artifacts.",
                ],
            )
        )

    if low_depth_categories:
        findings.append(
            _new_finding(
                finding_id="STEP-002",
                severity="P1",
                title="Step Library Depth Below Production Target",
                impact="Insufficient per-category step depth weakens fallback and escalation quality.",
                metrics={"categories_below_8_steps": len(low_depth_categories), "total_categories": len(steps_per_category)},
                evidence=low_depth_categories[:12],
                fix_plan=[
                    "Rebuild to 8-12 steps/category with minimum L1/L2/L3 structure retained.",
                    "Add fail-fast linter rule for minimum category depth in FULL mode.",
                ],
            )
        )

    if highly_reused_templates:
        findings.append(
            _new_finding(
                finding_id="STEP-003",
                severity="P1",
                title="High Template Reuse Across Categories",
                impact="Generic steps reduce category specificity and can mislead users with non-actionable guidance.",
                metrics={
                    "highly_reused_instruction_templates": len(highly_reused_templates),
                    "max_template_reuse": max(highly_reused_templates.values()) if highly_reused_templates else 0,
                },
                evidence=[f"reuse={count}" for _, count in sorted(highly_reused_templates.items(), key=lambda item: -item[1])[:8]],
                masked_samples=[sanitize_for_artifact(template, max_chars=220) for template in list(highly_reused_templates.keys())[:3]],
                fix_plan=[
                    "Generate category-specific instructions using telecom action lexicon constraints.",
                    "Add uniqueness threshold checks for instructions_tr and title_tr.",
                ],
            )
        )

    if action_coverage_missing > 0:
        findings.append(
            _new_finding(
                finding_id="STEP-004",
                severity="P1",
                title="Telecom Action Lexicon Coverage Gaps",
                impact="Steps may be too abstract and fail to provide operator-agnostic telecom troubleshooting actions.",
                metrics={
                    "steps_missing_action_lexicon": action_coverage_missing,
                    "step_count": len(steps),
                    "missing_rate": round(action_coverage_missing / len(steps), 6) if steps else 0.0,
                },
                evidence=["artifacts/solution_steps.jsonl"],
                masked_samples=action_missing_samples,
                fix_plan=[
                    "Require at least one telecom action token per step (APN, VoLTE, DNS, modem reset, roaming, etc.).",
                    "Reject steps failing action-verb linter in FULL mode.",
                ],
            )
        )

    if category_alignment_errors:
        findings.append(
            _new_finding(
                finding_id="STEP-005",
                severity="P1",
                title="Step ID and Category Alignment Errors",
                impact="Mismatched IDs can corrupt category routing and traceability.",
                metrics={"alignment_error_count": len(category_alignment_errors)},
                evidence=category_alignment_errors[:10],
                fix_plan=["Enforce STEP.<CATEGORY>.<NNN> and category_id consistency at generation time."],
            )
        )

    if lint_report["violations_count"] > 0 or legal_hits > 0 or guarantee_hits > 0 or operator_hits > 0:
        severity = "P0" if any(item["rule"] == "pii_request" for item in lint_report["violations"]) else "P1"
        findings.append(
            _new_finding(
                finding_id="STEP-006",
                severity=severity,
                title="Policy/Safety Lint Violations In Steps",
                impact="Safety and compliance constraints are not fully enforced in generated instructions.",
                metrics={
                    "linter_violations": lint_report["violations_count"],
                    "legal_hits": legal_hits,
                    "refund_guarantee_hits": guarantee_hits,
                    "operator_specific_hits": operator_hits,
                },
                evidence=[sanitize_for_artifact(json.dumps(item, ensure_ascii=False), max_chars=220) for item in lint_report["violations"][:8]],
                masked_samples=[sanitize_for_artifact(step.instructions_tr[0], max_chars=200) for step in steps[:5]],
                fix_plan=[
                    "Strengthen step linting to block PII requests, legal hallucinations, and guarantees.",
                    "Turn all safety lints into hard failures in FULL mode.",
                ],
            )
        )

    metrics = {
        "step_count": len(steps),
        "category_count": len(steps_per_category),
        "level_counts": dict(level_counts),
        "schema_error_count": len(step_errors),
        "duplicate_title_count": duplicate_title_count,
        "highly_reused_instruction_templates": len(highly_reused_templates),
        "steps_missing_action_lexicon": action_coverage_missing,
        "linter_violations": lint_report["violations_count"],
        "categories_below_8_steps": len(low_depth_categories),
    }
    report = _write_report(
        "solution_steps_report",
        metrics=metrics,
        findings=findings,
        scope=["artifacts/solution_steps.jsonl", "solution_steps/linting.py"],
    )
    return report, steps


def audit_kb(steps: list[SolutionStep]) -> tuple[dict[str, Any], list[KBParagraph]]:
    kb_path = PROJECT_ROOT / "artifacts" / "kb.jsonl"
    kb_rows, kb_errors = _load_jsonl_models(kb_path, KBParagraph)
    lint_report = lint_kb_paragraphs(kb_rows)

    text_counts = Counter(row.text_tr.strip().lower() for row in kb_rows)
    duplicate_text_rows = sum(count - 1 for count in text_counts.values() if count > 1)
    max_text_reuse = max(text_counts.values()) if text_counts else 0

    step_category = {step.step_id: step.category_id for step in steps}
    category_mismatch = 0
    mismatch_samples: list[str] = []
    reuse_by_text = defaultdict(set)
    pii_hits = Counter()
    pii_samples: list[str] = []
    legal_hits = 0
    generic_filler_hits = 0

    generic_markers = ["bu adim", "temel hedef", "teknik ekibin olayi", "guvenli bir inceleme akisina"]

    for row in kb_rows:
        doc_parts = row.doc_id.split(".")
        doc_category = doc_parts[1] if len(doc_parts) >= 3 else ""
        for step_id in row.applies_to_step_ids:
            step_cat = step_category.get(step_id)
            if step_cat is None:
                category_mismatch += 1
                if len(mismatch_samples) < 5:
                    mismatch_samples.append(f"{row.paragraph_id} -> missing step {step_id}")
                continue
            if step_cat != doc_category:
                category_mismatch += 1
                if len(mismatch_samples) < 5:
                    mismatch_samples.append(f"{row.paragraph_id}: doc={doc_category}, step={step_cat}")

        normalized_text = row.text_tr.strip().lower()
        reuse_by_text[normalized_text].add(doc_category)
        if any(marker in normalized_text for marker in generic_markers):
            generic_filler_hits += 1
        if LEGAL_PATTERN.search(row.text_tr):
            legal_hits += 1

        tags = detect_pii_tags(row.text_tr, ignore_mask_tokens=True)
        if tags:
            for tag in tags:
                pii_hits[tag] += 1
            if len(pii_samples) < 5:
                pii_samples.append(sanitize_for_artifact(row.text_tr, max_chars=220))

    cross_category_reuse = sum(1 for categories in reuse_by_text.values() if len(categories) > 1)
    unique_token_ratios = []
    for row in kb_rows:
        tokens = re.findall(r"[a-z0-9çğıöşü]+", row.text_tr.lower())
        if tokens:
            unique_token_ratios.append(len(set(tokens)) / float(len(tokens)))
    specificity_mean = round(mean(unique_token_ratios), 6) if unique_token_ratios else 0.0

    findings: list[dict[str, Any]] = []
    if kb_errors:
        findings.append(
            _new_finding(
                finding_id="KB-001",
                severity="P0",
                title="KB Schema Violations",
                impact="Invalid KB rows can break evidence linkage and runtime retrieval.",
                metrics={"schema_error_count": len(kb_errors)},
                evidence=kb_errors[:10],
                masked_samples=kb_errors[:5],
                fix_plan=[
                    "Fail KB stage when schema validation errors are present.",
                    "Add build-time schema test for KBParagraph artifacts.",
                ],
            )
        )

    if duplicate_text_rows > 0 or cross_category_reuse > 0:
        severity = "P0" if (duplicate_text_rows / max(1, len(kb_rows))) > 0.5 else "P1"
        findings.append(
            _new_finding(
                finding_id="KB-002",
                severity=severity,
                title="Heavy KB Paragraph Duplication",
                impact="Evidence diversity collapses; retrieval relevance and hallucination defense degrade.",
                metrics={
                    "duplicate_text_rows": duplicate_text_rows,
                    "cross_category_reuse_text_count": cross_category_reuse,
                    "max_text_reuse": max_text_reuse,
                },
                evidence=["artifacts/kb.jsonl"],
                masked_samples=[sanitize_for_artifact(item, max_chars=220) for item, count in text_counts.items() if count > 1][:5],
                fix_plan=[
                    "Rebuild KB with step-specific, category-specific paragraphs (1-3 unique paragraphs per step).",
                    "Enforce max reuse threshold and reject generic duplicates in FULL mode.",
                ],
            )
        )

    if category_mismatch > 0:
        findings.append(
            _new_finding(
                finding_id="KB-003",
                severity="P1",
                title="KB Category Alignment Mismatch",
                impact="Incorrect category linkage can attach irrelevant evidence to troubleshooting steps.",
                metrics={"category_mismatch_count": category_mismatch},
                evidence=mismatch_samples[:8],
                fix_plan=["Validate doc_id category and applies_to_step_ids category consistency pre-publish."],
            )
        )

    if generic_filler_hits > 0 and kb_rows and (generic_filler_hits / len(kb_rows)) > 0.3:
        findings.append(
            _new_finding(
                finding_id="KB-004",
                severity="P1",
                title="Generic KB Filler Text Overuse",
                impact="Content lacks actionable specificity and can cause repetitive low-value responses.",
                metrics={
                    "generic_filler_hits": generic_filler_hits,
                    "generic_filler_rate": round(generic_filler_hits / len(kb_rows), 6),
                    "specificity_score_mean": specificity_mean,
                },
                evidence=["artifacts/kb.jsonl"],
                masked_samples=[sanitize_for_artifact(row.text_tr, max_chars=220) for row in kb_rows[:5]],
                fix_plan=[
                    "Add specificity scoring and minimum threshold as linter gate.",
                    "Reject paragraph templates reused across unrelated categories.",
                ],
            )
        )

    if lint_report["violations_count"] > 0 or legal_hits > 0 or sum(pii_hits.values()) > 0:
        severity = "P0" if sum(pii_hits.values()) > 0 else "P1"
        findings.append(
            _new_finding(
                finding_id="KB-005",
                severity=severity,
                title="KB Safety/Compliance Lint Issues",
                impact="Unsafe or non-compliant evidence can leak into assistant responses.",
                metrics={
                    "kb_lint_violations": lint_report["violations_count"],
                    "legal_hits": legal_hits,
                    "pii_hits_total": sum(pii_hits.values()),
                    "pii_breakdown": dict(pii_hits),
                },
                evidence=[sanitize_for_artifact(json.dumps(item, ensure_ascii=False), max_chars=220) for item in lint_report["violations"][:8]],
                masked_samples=pii_samples[:5],
                fix_plan=[
                    "Apply strict KB safety linter and fail FULL mode on any PII or policy hallucination hit.",
                    "Add dedicated regression tests for forbidden content patterns.",
                ],
            )
        )

    metrics = {
        "kb_row_count": len(kb_rows),
        "schema_error_count": len(kb_errors),
        "duplicate_text_rows": duplicate_text_rows,
        "cross_category_reuse_text_count": cross_category_reuse,
        "max_text_reuse": max_text_reuse,
        "category_mismatch_count": category_mismatch,
        "generic_filler_hits": generic_filler_hits,
        "specificity_score_mean": specificity_mean,
        "kb_lint_violations": lint_report["violations_count"],
    }
    report = _write_report(
        "kb_report",
        metrics=metrics,
        findings=findings,
        scope=["artifacts/kb.jsonl", "solution_steps/linting.py"],
    )
    return report, kb_rows

def audit_step_kb_integrity(steps: list[SolutionStep], kb_rows: list[KBParagraph]) -> dict[str, Any]:
    links_path = PROJECT_ROOT / "artifacts" / "step_kb_links.jsonl"
    links, link_errors = _load_jsonl_models(links_path, StepKBLink)

    step_ids = {step.step_id for step in steps}
    kb_ids = {row.paragraph_id for row in kb_rows}
    step_category = {step.step_id: step.category_id for step in steps}

    dangling_steps = []
    dangling_evidence = []
    evidence_to_categories: dict[str, set[str]] = defaultdict(set)
    rationale_counts = Counter(link.rationale.strip().lower() for link in links)

    linked_steps = set()
    for link in links:
        linked_steps.add(link.step_id)
        if link.step_id not in step_ids:
            dangling_steps.append(link.step_id)
        for evidence_id in link.evidence_ids:
            if evidence_id not in kb_ids:
                dangling_evidence.append(f"{link.step_id}->{evidence_id}")
            if link.step_id in step_category:
                evidence_to_categories[evidence_id].add(step_category[link.step_id])

    steps_without_evidence = sorted(step_ids - linked_steps)
    unrelated_reuse = sorted([evidence_id for evidence_id, categories in evidence_to_categories.items() if len(categories) > 1])
    coverage = round((len(linked_steps) / len(step_ids)), 6) if step_ids else 0.0
    identical_rationale_count = max(rationale_counts.values()) if rationale_counts else 0

    findings: list[dict[str, Any]] = []
    if link_errors:
        findings.append(
            _new_finding(
                finding_id="LINK-001",
                severity="P0",
                title="Step-KB Link Schema Errors",
                impact="Broken link artifacts can invalidate retrieval evidence guarantees.",
                metrics={"schema_error_count": len(link_errors)},
                evidence=link_errors[:8],
                masked_samples=link_errors[:5],
                fix_plan=["Enforce StepKBLink schema validation before publishing link artifacts."],
            )
        )

    if dangling_steps or dangling_evidence:
        findings.append(
            _new_finding(
                finding_id="LINK-002",
                severity="P0",
                title="Dangling Step/Evidence References",
                impact="Dangling links break evidence traceability and may cause runtime validation failures.",
                metrics={"dangling_step_count": len(dangling_steps), "dangling_evidence_count": len(dangling_evidence)},
                evidence=(dangling_steps + dangling_evidence)[:10],
                fix_plan=[
                    "Reject link artifacts containing missing step_id or evidence_id references.",
                    "Add integrity tests to CI for every build.",
                ],
            )
        )

    if steps_without_evidence or coverage < 1.0:
        findings.append(
            _new_finding(
                finding_id="LINK-003",
                severity="P0",
                title="Incomplete Step-to-Evidence Coverage",
                impact="Some steps can be selected without evidence backing, increasing hallucination risk.",
                metrics={"step_coverage_ratio": coverage, "steps_without_link_count": len(steps_without_evidence)},
                evidence=steps_without_evidence[:10],
                fix_plan=["Require 100% step coverage with at least one evidence paragraph per step."],
            )
        )

    if unrelated_reuse:
        findings.append(
            _new_finding(
                finding_id="LINK-004",
                severity="P1",
                title="Evidence Reused Across Multiple Categories",
                impact="Cross-category evidence reuse can dilute category precision.",
                metrics={"cross_category_evidence_reuse_count": len(unrelated_reuse)},
                evidence=unrelated_reuse[:10],
                fix_plan=[
                    "Set max reuse threshold for evidence across unrelated categories.",
                    "Introduce category-specific rationale and evidence partitioning.",
                ],
            )
        )

    if identical_rationale_count > 3:
        findings.append(
            _new_finding(
                finding_id="LINK-005",
                severity="P2",
                title="Rationale Text Repetition",
                impact="Uniform rationale text reduces auditability and explainability depth.",
                metrics={"max_identical_rationale_reuse": identical_rationale_count},
                evidence=[f"unique_rationale_count={len(rationale_counts)}"],
                fix_plan=["Generate rationale fields based on step intent and evidence content."],
            )
        )

    metrics = {
        "link_row_count": len(links),
        "schema_error_count": len(link_errors),
        "step_coverage_ratio": coverage,
        "dangling_step_count": len(dangling_steps),
        "dangling_evidence_count": len(dangling_evidence),
        "cross_category_evidence_reuse_count": len(unrelated_reuse),
        "max_identical_rationale_reuse": identical_rationale_count,
    }
    return _write_report(
        "step_kb_integrity_report",
        metrics=metrics,
        findings=findings,
        scope=["artifacts/solution_steps.jsonl", "artifacts/kb.jsonl", "artifacts/step_kb_links.jsonl"],
    )


def audit_graph_integrity() -> dict[str, Any]:
    nodes_path = PROJECT_ROOT / "artifacts" / "graph" / "nodes.jsonl"
    edges_path = PROJECT_ROOT / "artifacts" / "graph" / "edges.jsonl"

    nodes, node_errors = _load_jsonl_models(nodes_path, GraphNode)
    edges, edge_errors = _load_jsonl_models(edges_path, GraphEdge)

    node_ids = {node.node_id for node in nodes}
    dangling_edges = []
    invalid_relation_shapes = []
    degree = Counter()
    invalid_weights = 0
    feature_missing = 0

    expected_by_type = {
        "complaint": {"brand", "time_bucket", "normalized_category", "category_confidence"},
        "solution_step": {"category_id", "level", "risk_level"},
        "kb_paragraph": {"doc_id", "confidence"},
    }
    relation_constraints = {
        "CLASSIFIED_AS": ("complaint", "category"),
        "HAS_BRAND": ("complaint", "brand"),
        "HAS_TIME_BUCKET": ("complaint", "time_bucket"),
        "RECOMMENDS_STEP": ("category", "solution_step"),
        "SUPPORTED_BY": ("solution_step", "kb_paragraph"),
    }
    node_type_by_id = {node.node_id: node.node_type for node in nodes}

    for node in nodes:
        expected = expected_by_type.get(node.node_type, set())
        if expected and not expected.issubset(set(node.attributes.keys())):
            feature_missing += 1

    for edge in edges:
        if edge.source_node_id not in node_ids or edge.target_node_id not in node_ids:
            dangling_edges.append(edge.edge_id)
        degree[edge.source_node_id] += 1
        degree[edge.target_node_id] += 1
        if not (0.0 <= edge.weight <= 1.0):
            invalid_weights += 1
        expected = relation_constraints.get(edge.relation_type)
        if expected is not None:
            src_type = node_type_by_id.get(edge.source_node_id)
            dst_type = node_type_by_id.get(edge.target_node_id)
            if (src_type, dst_type) != expected:
                invalid_relation_shapes.append(f"{edge.edge_id}:{src_type}->{dst_type}, expected {expected[0]}->{expected[1]}")

    isolated_nodes = [node.node_id for node in nodes if degree[node.node_id] == 0]

    findings: list[dict[str, Any]] = []
    if node_errors or edge_errors:
        findings.append(
            _new_finding(
                finding_id="GRAPH-001",
                severity="P0",
                title="Graph Artifact Schema Errors",
                impact="Invalid graph rows can break retrieval and graph traversal guarantees.",
                metrics={"node_schema_errors": len(node_errors), "edge_schema_errors": len(edge_errors)},
                evidence=(node_errors + edge_errors)[:10],
                masked_samples=(node_errors + edge_errors)[:5],
                fix_plan=[
                    "Validate node/edge schemas before graph publish.",
                    "Abort FULL mode when schema errors exist.",
                ],
            )
        )

    if dangling_edges:
        findings.append(
            _new_finding(
                finding_id="GRAPH-002",
                severity="P0",
                title="Dangling Graph Edges",
                impact="Dangling references cause retrieval inconsistency and potential runtime crashes.",
                metrics={"dangling_edge_count": len(dangling_edges)},
                evidence=dangling_edges[:10],
                fix_plan=["Enforce referential integrity checks between edge endpoints and node index."],
            )
        )

    if invalid_weights > 0:
        findings.append(
            _new_finding(
                finding_id="GRAPH-003",
                severity="P0",
                title="Invalid Edge Weights",
                impact="Weights outside [0,1] invalidate ranking assumptions.",
                metrics={"invalid_weight_count": invalid_weights},
                evidence=["artifacts/graph/edges.jsonl"],
                fix_plan=["Clamp/validate all edge weights and fail build on violations."],
            )
        )

    if isolated_nodes:
        findings.append(
            _new_finding(
                finding_id="GRAPH-004",
                severity="P1",
                title="Isolated Nodes Present",
                impact="Isolated nodes are dead graph branches that cannot contribute to retrieval.",
                metrics={"isolated_node_count": len(isolated_nodes)},
                evidence=isolated_nodes[:12],
                fix_plan=["Drop or reconnect isolated nodes during graph build."],
            )
        )

    if feature_missing > 0:
        findings.append(
            _new_finding(
                finding_id="GRAPH-005",
                severity="P1",
                title="Missing Expected Node Features",
                impact="Missing attributes weaken explainability and downstream feature usage.",
                metrics={"nodes_missing_expected_features": feature_missing},
                evidence=["artifacts/graph/nodes.jsonl"],
                fix_plan=["Add node-type-specific feature contracts and schema tests."],
            )
        )

    if invalid_relation_shapes:
        findings.append(
            _new_finding(
                finding_id="GRAPH-006",
                severity="P1",
                title="Relation Type Shape Violations",
                impact="Unexpected source/target node types reduce graph semantic consistency.",
                metrics={"relation_shape_violation_count": len(invalid_relation_shapes)},
                evidence=invalid_relation_shapes[:10],
                fix_plan=["Validate relation_type allowed endpoint node types in graph builder."],
            )
        )

    metrics = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_schema_errors": len(node_errors),
        "edge_schema_errors": len(edge_errors),
        "dangling_edge_count": len(dangling_edges),
        "isolated_node_count": len(isolated_nodes),
        "invalid_weight_count": invalid_weights,
        "nodes_missing_expected_features": feature_missing,
        "relation_shape_violation_count": len(invalid_relation_shapes),
    }
    return _write_report(
        "graph_integrity_report",
        metrics=metrics,
        findings=findings,
        scope=["artifacts/graph/nodes.jsonl", "artifacts/graph/edges.jsonl", "graph/builder.py"],
    )

def audit_eval_integrity(config: dict[str, Any]) -> dict[str, Any]:
    eval_dir = PROJECT_ROOT / "artifacts" / "eval"
    combined_dashboard_path = eval_dir / "combined_dashboard.json"
    hallucination_path = eval_dir / "hallucination_report.json"
    pii_path = eval_dir / "pii_leak_report.json"
    security_path = eval_dir / "security_adversarial_report.json"
    task_path = eval_dir / "task_metrics_report.json"
    retrieval_eval_path = PROJECT_ROOT / "artifacts" / "retrieval_eval.json"

    eval_payloads: dict[str, dict[str, Any] | None] = {}
    for key, path in [
        ("combined_dashboard", combined_dashboard_path),
        ("hallucination_report", hallucination_path),
        ("pii_report", pii_path),
        ("security_report", security_path),
        ("task_report", task_path),
        ("retrieval_eval", retrieval_eval_path),
    ]:
        if path.exists():
            try:
                eval_payloads[key] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                eval_payloads[key] = None
        else:
            eval_payloads[key] = None

    combined_dashboard = eval_payloads.get("combined_dashboard") or {}
    key_metrics = combined_dashboard.get("key_metrics", {})

    hallucination_source = _read_text(PROJECT_ROOT / "evaluation" / "hallucination.py")
    eval_common_source = _read_text(PROJECT_ROOT / "evaluation" / "common.py")
    infer_source = _read_text(PROJECT_ROOT / "models" / "infer.py")
    lora_source = _read_text(PROJECT_ROOT / "training" / "lora_trainer.py")

    has_error_fallback = "error_fallback" in eval_common_source
    has_renderer_fallback = "renderer_fallback" in infer_source
    has_mock_fallback = "fallback_to_mock_on_failure" in lora_source and "mock" in lora_source

    has_semantic_mismatch_check = ("cosine" in hallucination_source.lower()) or ("similarity" in hallucination_source.lower())
    fail_fast_cfg = config.get("fail_fast")
    has_required_fail_fast_block = isinstance(fail_fast_cfg, dict) and all(
        key in fail_fast_cfg
        for key in [
            "schema_violation",
            "pii_leak",
            "hallucination_violation",
            "missing_evidence",
            "graph_inconsistency",
        ]
    )

    model_artifacts = []
    models_root = PROJECT_ROOT / "artifacts" / "models"
    if models_root.exists():
        for config_file in sorted(models_root.glob("*/adapter_config.json")):
            try:
                payload = json.loads(config_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            model_artifacts.append(
                {
                    "run_id": payload.get("run_id"),
                    "backend": payload.get("backend"),
                    "status": payload.get("status"),
                    "note": payload.get("metrics", {}).get("note"),
                }
            )

    findings: list[dict[str, Any]] = []

    if has_error_fallback:
        findings.append(
            _new_finding(
                finding_id="EVAL-001",
                severity="P0",
                title="Evaluation Uses Error Fallback Outputs",
                impact="Inference failures can be converted into synthetic responses, masking real reliability issues.",
                metrics={"error_fallback_present": True},
                evidence=["evaluation/common.py contains generation_mode=error_fallback"],
                fix_plan=[
                    "Remove synthetic success fallback in FULL mode; fail case explicitly.",
                    "Track inference failures as hard gate metrics.",
                ],
            )
        )

    if has_renderer_fallback:
        findings.append(
            _new_finding(
                finding_id="EVAL-002",
                severity="P0",
                title="Renderer Fallback Can Mask Model Behavior",
                impact="Hallucination/security metrics may reflect templated fallback quality instead of actual model quality.",
                metrics={"renderer_fallback_present": True},
                evidence=["models/infer.py uses renderer_fallback when model path fails"],
                fix_plan=[
                    "Disable renderer fallback in FULL mode and fail-fast on backend failures.",
                    "Separate model-only evaluation from template fallback smoke checks.",
                ],
            )
        )

    if not has_required_fail_fast_block:
        findings.append(
            _new_finding(
                finding_id="EVAL-003",
                severity="P0",
                title="Missing Global fail_fast Safety Gate Config",
                impact="Required schema/PII/hallucination/graph safety gates are not centrally enforced.",
                metrics={"has_required_fail_fast_block": False},
                evidence=["config.yaml does not define config.fail_fast with required keys"],
                fix_plan=[
                    "Add config.fail_fast block with mandatory booleans and enforce in all stages.",
                    "Emit artifacts/aborted_reason.json on any FULL mode violation.",
                ],
            )
        )

    if not has_semantic_mismatch_check:
        findings.append(
            _new_finding(
                finding_id="EVAL-004",
                severity="P1",
                title="Hallucination Scorer Lacks Semantic Evidence Mismatch Check",
                impact="ID-level checks alone can miss semantically wrong claims with valid-looking citations.",
                metrics={"semantic_mismatch_check_present": False},
                evidence=["evaluation/hallucination.py checks IDs but no semantic similarity layer"],
                fix_plan=[
                    "Add semantic similarity validation between actionable claims and cited evidence text.",
                    "Introduce actionable-claim-without-citation and contradiction checks as hard gates.",
                ],
            )
        )

    if has_mock_fallback:
        findings.append(
            _new_finding(
                finding_id="EVAL-005",
                severity="P1",
                title="Training Pipeline Allows Mock Fallback",
                impact="Model artifact status can appear completed without real training.",
                metrics={"mock_fallback_config_present": True},
                evidence=["training/lora_trainer.py includes fallback_to_mock_on_failure path"],
                fix_plan=[
                    "Disable mock fallback in FULL mode.",
                    "Require backend=transformers_peft (or approved real backend) for production readiness.",
                ],
            )
        )

    macro_f1 = key_metrics.get("intent_macro_f1")
    step_validity = key_metrics.get("step_validity_rate")
    if isinstance(macro_f1, (int, float)) and isinstance(step_validity, (int, float)):
        if macro_f1 >= 0.99 and step_validity >= 0.99:
            findings.append(
                _new_finding(
                    finding_id="EVAL-006",
                    severity="P2",
                    title="Potential Metric Gamability Signal",
                    impact="Near-perfect metrics combined with fallback paths suggests evaluation may be optimistic.",
                    metrics={"intent_macro_f1": macro_f1, "step_validity_rate": step_validity},
                    evidence=["artifacts/eval/combined_dashboard.json"],
                    fix_plan=[
                        "Add adversarial hard sets and model-only scoring path without templated fallback.",
                        "Gate on robustness metrics, not only aggregate averages.",
                    ],
                )
            )

    mock_runs = [item for item in model_artifacts if item.get("backend") == "mock"]
    if mock_runs:
        findings.append(
            _new_finding(
                finding_id="EVAL-007",
                severity="P1",
                title="Model Artifacts Indicate Mock Backend",
                impact="Current artifacts do not represent trained LLM behavior.",
                metrics={"mock_model_run_count": len(mock_runs), "total_model_runs": len(model_artifacts)},
                evidence=[sanitize_for_artifact(json.dumps(item, ensure_ascii=False), max_chars=220) for item in mock_runs[:5]],
                fix_plan=["Mark mock-backed runs as non-production and exclude them from readiness dashboards."],
            )
        )

    metrics = {
        "eval_artifacts_present": sum(1 for value in eval_payloads.values() if value is not None),
        "combined_dashboard_overall_pass": combined_dashboard.get("overall_pass"),
        "intent_macro_f1": key_metrics.get("intent_macro_f1"),
        "step_validity_rate": key_metrics.get("step_validity_rate"),
        "hallucination_rate_actionable": key_metrics.get("hallucination_rate_actionable"),
        "renderer_fallback_present": has_renderer_fallback,
        "error_fallback_present": has_error_fallback,
        "has_required_fail_fast_block": has_required_fail_fast_block,
        "mock_model_runs": len(mock_runs),
    }
    return _write_report(
        "eval_integrity_report",
        metrics=metrics,
        findings=findings,
        scope=[
            "evaluation/*.py",
            "models/infer.py",
            "training/lora_trainer.py",
            "artifacts/eval/*",
            "artifacts/retrieval_eval.json",
            "artifacts/models/*",
        ],
    )


def run_phase1_audit() -> dict[str, Any]:
    config = _load_config(PROJECT_ROOT / "config.yaml")
    reports: dict[str, Any] = {}

    reports["code"] = audit_code_quality(config)
    data_report, _, labeled_rows = audit_data_quality()
    reports["data"] = data_report
    reports["taxonomy"] = audit_taxonomy(labeled_rows)
    solution_report, steps = audit_solution_steps()
    reports["solution_steps"] = solution_report
    kb_report, kb_rows = audit_kb(steps)
    reports["kb"] = kb_report
    reports["step_kb_integrity"] = audit_step_kb_integrity(steps, kb_rows)
    reports["graph"] = audit_graph_integrity()
    reports["eval"] = audit_eval_integrity(config)

    p0 = sum(item["severity_counts"]["P0"] for item in reports.values())
    p1 = sum(item["severity_counts"]["P1"] for item in reports.values())
    p2 = sum(item["severity_counts"]["P2"] for item in reports.values())
    summary = {
        "generated_at_utc": _now_iso(),
        "reports": {name: report["overall_status"] for name, report in reports.items()},
        "severity_totals": {"P0": p0, "P1": p1, "P2": p2},
    }
    (AUDIT_DIR / "phase1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase-1 full audit generator")
    parser.parse_args()
    summary = run_phase1_audit()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
