from __future__ import annotations

from preprocess.pii import mask_pii_text


def test_pii_masking_replaces_sensitive_tokens() -> None:
    text = (
        "Telefon numaram 0532 123 45 67, e-posta adresim ali.kaya@example.com, "
        "IBAN TR330006100519786457841326, TCKN 12345678901 ve musteri no: 998877."
    )
    result = mask_pii_text(text)

    assert "[PHONE]" in result.masked_text
    assert "[EMAIL]" in result.masked_text
    assert "[IBAN]" in result.masked_text
    assert "[TCKN]" in result.masked_text
    assert "[ACCOUNT_ID]" in result.masked_text

    assert "0532 123 45 67" not in result.masked_text
    assert "ali.kaya@example.com" not in result.masked_text
    assert "TR330006100519786457841326" not in result.masked_text
    assert "12345678901" not in result.masked_text
    assert not result.remaining_tags

