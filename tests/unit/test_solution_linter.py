from __future__ import annotations

from data.schemas import KBParagraph, SolutionStep
from solution_steps.linting import lint_kb_paragraphs, lint_solution_steps


def test_linter_catches_forbidden_pii_and_promise_patterns() -> None:
    bad_step = SolutionStep(
        step_id="STEP.BILLING_PAYMENTS.001",
        category_id="BILLING_PAYMENTS",
        level="L1",
        title_tr="Fatura adimi",
        instructions_tr=[
            "Lutfen TCKN ve IBAN bilgilerinizi iletin.",
            "Islem bugun duzelir ve kesin iade verilir.",
            "Sonucu bekleyin.",
        ],
        required_inputs=["time_window"],
        success_check="Sonuc alinir.",
        stop_conditions=["Sorun devam ederse destek al."],
        escalation_unit="BILLING_SUPPORT",
        risk_level="low",
        tags=[],
        version="solution-steps-v1",
    )
    step_report = lint_solution_steps([bad_step])
    assert step_report["violations_count"] >= 2

    bad_kb = KBParagraph(
        doc_id="KB.BILLING_PAYMENTS.001",
        paragraph_id="KB.BILLING_PAYMENTS.001#P1",
        text_tr="Kanun no 9999 geregi bugun kesin duzelir.",
        applies_to_step_ids=["STEP.BILLING_PAYMENTS.001"],
        source_type="internal_best_practice",
        confidence=0.7,
        version="solution-steps-v1",
    )
    kb_report = lint_kb_paragraphs([bad_kb])
    assert kb_report["violations_count"] >= 1

