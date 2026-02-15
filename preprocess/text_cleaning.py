from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import dataclass


DEFAULT_SCRIPT_INDICATORS = [
    "<script",
    "googletag",
    "defineslot",
    "adsdata",
    "prebid",
    "pubads",
    "gpt_unit_",
    "window.addeventlistener",
    "googletagmanager",
]

INLINE_SCRIPT_TAG_PATTERN = re.compile(r"(?is)<script[^>]*>.*?</script>")
HTML_TAG_PATTERN = re.compile(r"(?is)<[^>]+>")
JS_BLOB_PATTERN = re.compile(
    r"(?i)\b(?:function|var|const|let|window\.|document\.|googletag|prebid|pubads|adsdata|defineSlot)\b"
)


@dataclass(frozen=True)
class ScriptNoiseAssessment:
    indicator_hits: int
    js_line_ratio: float
    alpha_ratio: float
    cleaned_ratio: float
    mostly_noise: bool


@dataclass(frozen=True)
class MultiComplaintAssessment:
    suspected: bool
    ellipsis_count: int
    repeated_sikayet_count: int
    repeated_brand_header_count: int
    short_paragraph_ratio: float
    split_candidates: list[str]


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return text.strip()


def _remove_repeated_lines(lines: list[str], keep_repeats: int = 2) -> list[str]:
    counts: Counter[str] = Counter()
    kept: list[str] = []
    for line in lines:
        normalized = re.sub(r"\s+", " ", line.strip().lower())
        counts[normalized] += 1
        if counts[normalized] <= keep_repeats:
            kept.append(line)
    return kept


def clean_text_content(text: str, indicators: list[str] | None = None) -> tuple[str, dict[str, int]]:
    active_indicators = indicators or DEFAULT_SCRIPT_INDICATORS
    lowered_indicators = [item.lower() for item in active_indicators]

    cleaned = html.unescape(text or "")
    cleaned = INLINE_SCRIPT_TAG_PATTERN.sub(" ", cleaned)

    dropped_script_lines = 0
    retained_lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line_lower = line.lower()
        has_indicator = any(ind in line_lower for ind in lowered_indicators)
        has_js_blob = bool(JS_BLOB_PATTERN.search(line)) and (";" in line or "{" in line or "}" in line)
        if has_indicator or has_js_blob:
            dropped_script_lines += 1
            continue
        retained_lines.append(line)

    retained_lines = _remove_repeated_lines(retained_lines)
    cleaned = "\n".join(retained_lines)
    cleaned = HTML_TAG_PATTERN.sub(" ", cleaned)
    cleaned = _normalize_whitespace(cleaned)

    stats = {
        "dropped_script_lines": dropped_script_lines,
        "retained_line_count": len(retained_lines),
    }
    return cleaned, stats


def assess_script_noise(
    raw_text: str,
    cleaned_text: str,
    indicators: list[str] | None = None,
    min_indicator_hits: int = 2,
    min_js_line_ratio: float = 0.35,
    min_alpha_ratio: float = 0.25,
    min_cleaned_ratio: float = 0.2,
) -> ScriptNoiseAssessment:
    active_indicators = indicators or DEFAULT_SCRIPT_INDICATORS
    lower_raw = (raw_text or "").lower()
    indicator_hits = sum(lower_raw.count(ind.lower()) for ind in active_indicators)

    lines = [line.strip() for line in (raw_text or "").splitlines() if line.strip()]
    if lines:
        js_lines = [
            line
            for line in lines
            if JS_BLOB_PATTERN.search(line)
            or any(ind.lower() in line.lower() for ind in active_indicators)
            or line.startswith("<script")
        ]
        js_line_ratio = len(js_lines) / len(lines)
    else:
        js_line_ratio = 0.0

    raw_non_space = re.sub(r"\s+", "", raw_text or "")
    alpha_chars = sum(1 for char in raw_non_space if char.isalpha())
    alpha_ratio = (alpha_chars / len(raw_non_space)) if raw_non_space else 0.0
    cleaned_ratio = (len(cleaned_text) / len(raw_text)) if raw_text else 0.0

    mostly_noise = indicator_hits >= min_indicator_hits and (
        js_line_ratio >= min_js_line_ratio or alpha_ratio < min_alpha_ratio or cleaned_ratio < min_cleaned_ratio
    )

    return ScriptNoiseAssessment(
        indicator_hits=indicator_hits,
        js_line_ratio=round(js_line_ratio, 4),
        alpha_ratio=round(alpha_ratio, 4),
        cleaned_ratio=round(cleaned_ratio, 4),
        mostly_noise=mostly_noise,
    )


def assess_multi_complaint(text: str, brand_name: str | None = None) -> MultiComplaintAssessment:
    normalized = text or ""
    ellipsis_count = len(re.findall(r"\.\.\.", normalized))
    repeated_sikayet_count = len(re.findall(r"(?iu)\bşikayet\b", normalized))

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    repeated_brand_header_count = 0
    if brand_name:
        name = brand_name.lower()
        repeated_brand_header_count = sum(1 for line in lines if line.lower().startswith(name))

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", normalized) if part.strip()]
    short_paragraphs = [paragraph for paragraph in paragraphs if len(paragraph) < 120]
    short_paragraph_ratio = (len(short_paragraphs) / len(paragraphs)) if paragraphs else 0.0

    split_candidates = split_multi_complaint_blocks(normalized, brand_name=brand_name)
    suspected = (
        (ellipsis_count >= 4 and len(split_candidates) >= 2)
        or repeated_brand_header_count >= 2
        or (repeated_sikayet_count >= 3 and short_paragraph_ratio >= 0.6 and len(paragraphs) >= 6)
    )

    return MultiComplaintAssessment(
        suspected=suspected,
        ellipsis_count=ellipsis_count,
        repeated_sikayet_count=repeated_sikayet_count,
        repeated_brand_header_count=repeated_brand_header_count,
        short_paragraph_ratio=round(short_paragraph_ratio, 4),
        split_candidates=split_candidates,
    )


def split_multi_complaint_blocks(text: str, brand_name: str | None = None) -> list[str]:
    candidate = text or ""
    if not candidate:
        return []

    segments = [part.strip() for part in re.split(r"(?:\n?\s*\.\.\.\s*\n?)+", candidate) if part.strip()]
    if brand_name:
        name = re.escape(brand_name)
        split_by_brand = []
        for segment in segments:
            pieces = re.split(rf"(?im)^(?={name}\b)", segment)
            split_by_brand.extend(piece.strip() for piece in pieces if piece.strip())
        segments = split_by_brand if split_by_brand else segments

    deduped: list[str] = []
    seen = set()
    for segment in segments:
        normalized = re.sub(r"\s+", " ", segment.lower())
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(segment)
    return deduped


def extract_primary_complaint(text: str, brand_name: str | None = None, min_chars: int = 80) -> str:
    assessment = assess_multi_complaint(text=text, brand_name=brand_name)
    if not assessment.suspected:
        return (text or "").strip()

    candidates = assessment.split_candidates or [(text or "").strip()]
    for candidate in candidates:
        normalized = _normalize_whitespace(candidate)
        if len(normalized) >= min_chars:
            return normalized

    fallback = _normalize_whitespace(candidates[0]) if candidates else _normalize_whitespace(text or "")
    return fallback

