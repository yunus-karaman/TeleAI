from __future__ import annotations

import json
from pathlib import Path

from scripts.config_loader import load_config
from scripts.solution_dataset_integrity import validate_solution_dataset


def test_solution_dataset_integrity_passes_on_canonical_artifacts() -> None:
    config = load_config("config.yaml", "FULL")
    report = validate_solution_dataset(
        taxonomy_path=Path(config["taxonomy"]["taxonomy_file"]),
        solution_steps_path=Path(config["paths"]["solution_steps_jsonl"]),
        kb_path=Path(config["paths"]["kb_jsonl"]),
        step_kb_links_path=Path(config["paths"]["step_kb_links_jsonl"]),
        stage="unit_test_solution_dataset_integrity",
    )
    assert report["overall_pass"] is True
    assert report["counts"]["step_schema_errors"] == 0
    assert report["counts"]["kb_schema_errors"] == 0
    assert report["counts"]["link_schema_errors"] == 0


def test_solution_dataset_integrity_detects_step_without_evidence(tmp_path: Path) -> None:
    config = load_config("config.yaml", "FULL")
    taxonomy_path = Path(config["taxonomy"]["taxonomy_file"])
    steps_path = Path(config["paths"]["solution_steps_jsonl"])
    kb_path = Path(config["paths"]["kb_jsonl"])
    links_src = Path(config["paths"]["step_kb_links_jsonl"])

    link_rows = []
    for line in links_src.read_text(encoding="utf-8").splitlines():
        if line.strip():
            link_rows.append(json.loads(line))
    assert link_rows

    mutated = list(link_rows)
    mutated.pop(0)
    broken_links = tmp_path / "step_kb_links_broken.jsonl"
    broken_links.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in mutated), encoding="utf-8")

    report = validate_solution_dataset(
        taxonomy_path=taxonomy_path,
        solution_steps_path=steps_path,
        kb_path=kb_path,
        step_kb_links_path=broken_links,
        stage="unit_test_solution_dataset_integrity_broken",
    )
    assert report["overall_pass"] is False
    assert any(item["code"] == "STEP_WITHOUT_EVIDENCE" for item in report["violations"])
