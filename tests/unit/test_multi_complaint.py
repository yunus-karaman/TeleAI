from __future__ import annotations

from preprocess.text_cleaning import assess_multi_complaint, extract_primary_complaint


def test_multi_complaint_detection_and_primary_extraction() -> None:
    text = (
        "Bimcell hattimda internetim surekli kopuyor ve destek alamiyorum. Sorunun cozumu icin yardim istiyorum.\n"
        "...\n"
        "Bimcell Musteri Hizmetleri: Farkli bir kullaniciya ait ikinci sikayet parcasi burada tekrar ediyor.\n"
        "...\n"
        "Bimcell ucuncu snippet: Baska bir sikayet metni de buraya eklenmis."
    )

    assessment = assess_multi_complaint(text=text, brand_name="Bimcell")
    assert assessment.suspected is True

    primary = extract_primary_complaint(text=text, brand_name="Bimcell", min_chars=60)
    assert "internetim surekli kopuyor" in primary
    assert "ikinci sikayet parcasi" not in primary

