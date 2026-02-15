from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


MASK_TOKEN_PATTERN = re.compile(r"\[(?:PHONE|EMAIL|IBAN|TCKN|DEVICE_ID|ACCOUNT_ID|ADDRESS|ADDR_NO)\]")

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
IBAN_PATTERN = re.compile(r"\bTR\d{2}(?:\s?\d{4}){5}\s?\d{2}\b", flags=re.IGNORECASE)
PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:(?:\+?90|0090|0)\s*)?(?:\(?5\d{2}\)?[\s.\-]*)\d{3}[\s.\-]*\d{2}[\s.\-]*\d{2}(?!\w)"
)
TCKN_PATTERN = re.compile(r"(?<!\d)\d{11}(?!\d)")
LONG_DEVICE_NUMBER_PATTERN = re.compile(r"(?<!\d)\d{15,22}(?!\d)")
DEVICE_KEYWORD_PATTERN = re.compile(
    r"\b(?:imei|iccid|imsi|sim(?:\s*kart)?\s*no(?:su)?|cihaz\s*id|device\s*id|eid)\b\s*[:#-]?\s*[A-Z0-9]{6,22}",
    flags=re.IGNORECASE,
)
ACCOUNT_ID_PATTERN = re.compile(
    r"\b(?P<key>(?:m[uü]steri|abone(?:lik)?|hat|hesap)\s*no(?:su)?)\b\s*[:#-]?\s*(?P<id>[A-Z0-9\-]{3,})",
    flags=re.IGNORECASE,
)
ADDRESS_NUMBER_PATTERN = re.compile(r"\b(?P<prefix>no)\s*[:.]?\s*(?P<num>\d+[A-Za-z]?)\b", flags=re.IGNORECASE)
ADDRESS_CUE_PATTERN = re.compile(
    r"\b(?:mah(?:allesi)?\.?|cadde|cd\.?|sokak|sk\.?|daire|kat|apt\.?|apartman[ıi]?)\b(?:\s+[A-Za-zÇĞİÖŞÜçğıöşü0-9.\-]+){0,3}\s+\d+[A-Za-z]?\b",
    flags=re.IGNORECASE,
)


PII_DETECTION_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "EMAIL": [EMAIL_PATTERN],
    "IBAN": [IBAN_PATTERN],
    "PHONE": [PHONE_PATTERN],
    "TCKN": [TCKN_PATTERN],
    "DEVICE_ID": [DEVICE_KEYWORD_PATTERN, LONG_DEVICE_NUMBER_PATTERN],
    "ACCOUNT_ID": [ACCOUNT_ID_PATTERN],
    "ADDRESS": [ADDRESS_NUMBER_PATTERN, ADDRESS_CUE_PATTERN],
}


@dataclass(frozen=True)
class PIIMaskResult:
    masked_text: str
    detected_tags: list[str]
    remaining_tags: list[str]

    @property
    def had_pii(self) -> bool:
        return len(self.detected_tags) > 0


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return text.strip()


def detect_pii_tags(text: str, ignore_mask_tokens: bool = True) -> list[str]:
    candidate = text or ""
    if ignore_mask_tokens:
        candidate = MASK_TOKEN_PATTERN.sub(" ", candidate)

    detected: list[str] = []
    for tag, patterns in PII_DETECTION_PATTERNS.items():
        if any(pattern.search(candidate) for pattern in patterns):
            detected.append(tag)
    return sorted(set(detected))


def mask_pii_text(text: str) -> PIIMaskResult:
    masked = text or ""
    detected = detect_pii_tags(masked, ignore_mask_tokens=True)

    masked = EMAIL_PATTERN.sub("[EMAIL]", masked)
    masked = IBAN_PATTERN.sub("[IBAN]", masked)
    masked = DEVICE_KEYWORD_PATTERN.sub("[DEVICE_ID]", masked)
    masked = ACCOUNT_ID_PATTERN.sub(lambda match: f"{match.group('key')} [ACCOUNT_ID]", masked)
    masked = PHONE_PATTERN.sub("[PHONE]", masked)
    masked = TCKN_PATTERN.sub("[TCKN]", masked)
    masked = LONG_DEVICE_NUMBER_PATTERN.sub("[DEVICE_ID]", masked)
    masked = ADDRESS_NUMBER_PATTERN.sub(lambda match: f"{match.group('prefix')}: [ADDR_NO]", masked)
    masked = ADDRESS_CUE_PATTERN.sub("[ADDRESS]", masked)
    masked = re.sub(r"(?i)\[ADDR_NO\]\s*/\s*\d+\b", "[ADDR_NO]", masked)
    masked = _normalize_whitespace(masked)

    remaining = detect_pii_tags(masked, ignore_mask_tokens=True)
    return PIIMaskResult(
        masked_text=masked,
        detected_tags=detected,
        remaining_tags=remaining,
    )


def sanitize_for_artifact(value: Any, max_chars: int = 1600) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    masked = mask_pii_text(text)
    sanitized = masked.masked_text
    if masked.remaining_tags:
        sanitized = EMAIL_PATTERN.sub("[EMAIL]", sanitized)
        sanitized = IBAN_PATTERN.sub("[IBAN]", sanitized)
        sanitized = PHONE_PATTERN.sub("[PHONE]", sanitized)
        sanitized = TCKN_PATTERN.sub("[TCKN]", sanitized)
        sanitized = LONG_DEVICE_NUMBER_PATTERN.sub("[DEVICE_ID]", sanitized)
        sanitized = ACCOUNT_ID_PATTERN.sub(lambda match: f"{match.group('key')} [ACCOUNT_ID]", sanitized)
        sanitized = ADDRESS_NUMBER_PATTERN.sub(lambda match: f"{match.group('prefix')}: [ADDR_NO]", sanitized)
        sanitized = ADDRESS_CUE_PATTERN.sub("[ADDRESS]", sanitized)
        sanitized = re.sub(r"(?<!\w)\d{2,}(?!\w)", "[NUM]", sanitized)
    if len(sanitized) > max_chars:
        return sanitized[:max_chars] + "... [TRUNCATED]"
    return sanitized
