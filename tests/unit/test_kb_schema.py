from __future__ import annotations

from data.schemas import KBParagraph


def test_kb_paragraph_schema_valid_record() -> None:
    paragraph = KBParagraph(
        doc_id="KB.BILLING_PAYMENTS.001",
        paragraph_id="KB.BILLING_PAYMENTS.001#P1",
        text_tr=(
            "Bu adim fatura kaynakli farklari netlestirmek icin kullanilir. "
            "Kayitli test zamani ve hata semptomu destek ekibinin incelemesini hizlandirir."
        ),
        applies_to_step_ids=["STEP.BILLING_PAYMENTS.001"],
        source_type="internal_best_practice",
        confidence=0.82,
        version="solution-steps-v1",
    )
    assert paragraph.source_type == "internal_best_practice"

