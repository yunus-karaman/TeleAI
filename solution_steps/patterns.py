from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from data.schemas import NormalizedComplaint
from taxonomy.schema import TaxonomyFile


TOKEN_PATTERN = re.compile(r"[a-z0-9çğıöşü]{2,}", flags=re.IGNORECASE)
MASK_TOKEN_PATTERN = re.compile(r"\[[A-Z_]+\]")

COMMON_STOPWORDS = {
    "ve",
    "bir",
    "bu",
    "da",
    "de",
    "ile",
    "icin",
    "için",
    "gibi",
    "ama",
    "ancak",
    "sonra",
    "kadar",
    "ben",
    "bana",
    "olarak",
    "olan",
    "oldu",
    "oluyor",
    "fazla",
    "hala",
    "halaa",
    "cok",
    "çok",
    "talep",
    "ediyorum",
    "yaklasik",
    "yaklaşık",
    "saygilarimizla",
    "saygılarımızla",
    "degerli",
    "değerli",
    "musterimiz",
    "müşterimiz",
    "memnuniyet",
    "merkezi",
    "iyi",
    "gunler",
    "günler",
}

BOILERPLATE_PHRASES = [
    "saygilarimizla turkcell",
    "saygılarımızla turkcell",
    "vodafone memnuniyet merkezi",
    "degerli musterimiz",
    "değerli müşterimiz",
    "iyi gunler dileriz",
    "iyi günler dileriz",
    "talep ediyorum",
    "geri donus saglanacaktir",
]

OPERATOR_TERMS = {"turkcell", "vodafone", "turknet", "turk", "telekom", "bimcell", "pttcell"}

SYMPTOM_STEMS = [
    "hiz",
    "yavas",
    "kesinti",
    "ariza",
    "cekim",
    "sinyal",
    "fatura",
    "borc",
    "odeme",
    "sms",
    "arama",
    "internet",
    "paket",
    "iptal",
    "cayma",
    "tasima",
    "roaming",
    "kurulum",
    "altyapi",
    "modem",
    "giris",
    "hata",
    "aktivasyon",
]
CONTEXT_STEMS = [
    "modem",
    "router",
    "wifi",
    "android",
    "ios",
    "uygulama",
    "city",
    "sehir",
    "bolge",
    "bina",
    "ev",
    "isyeri",
    "4g",
    "5g",
    "4.5g",
    "fiber",
    "vdsl",
    "adsl",
    "sim",
    "hat",
]
TRIGGER_STEMS = [
    "sonrasi",
    "yenile",
    "guncelle",
    "tasima",
    "kurulum",
    "odeme",
    "aktivasyon",
    "taahhut",
    "iptal",
    "degisiklik",
    "sim",
    "paket",
]


def _normalize_text(text: str) -> str:
    normalized = (text or "").lower()
    normalized = MASK_TOKEN_PATTERN.sub(" ", normalized)
    for phrase in BOILERPLATE_PHRASES:
        normalized = normalized.replace(phrase, " ")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(_normalize_text(text))


def _valid_term(term: str) -> bool:
    if len(term) < 3:
        return False
    if term in COMMON_STOPWORDS or term in OPERATOR_TERMS:
        return False
    if term.isnumeric():
        return False
    return True


def _collect_term_counts(texts: list[str]) -> tuple[Counter[str], Counter[str]]:
    unigram_counts: Counter[str] = Counter()
    bigram_counts: Counter[str] = Counter()
    for text in texts:
        tokens = [token for token in _tokenize(text) if _valid_term(token)]
        for token in tokens:
            unigram_counts[token] += 1
        for index in range(len(tokens) - 1):
            bigram = f"{tokens[index]} {tokens[index + 1]}"
            if any(term in OPERATOR_TERMS for term in bigram.split()):
                continue
            bigram_counts[bigram] += 1
    return unigram_counts, bigram_counts


def _select_terms(
    unigrams: Counter[str],
    bigrams: Counter[str],
    stems: list[str],
    fallback_terms: list[str],
    top_k: int,
) -> list[str]:
    ranked: list[tuple[int, str]] = []
    for term, count in bigrams.items():
        if count < 2:
            continue
        if any(stem in term for stem in stems):
            ranked.append((count, term))
    for term, count in unigrams.items():
        if count < 2:
            continue
        if any(stem in term for stem in stems):
            ranked.append((count, term))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected: list[str] = []
    seen = set()
    for _, term in ranked:
        if term in seen:
            continue
        seen.add(term)
        selected.append(term)
        if len(selected) >= top_k:
            break

    for fallback in fallback_terms:
        fallback_clean = fallback.lower().strip()
        if not fallback_clean or fallback_clean in seen:
            continue
        selected.append(fallback_clean)
        seen.add(fallback_clean)
        if len(selected) >= top_k:
            break

    return selected[:top_k]


def _masked_complaint_id(raw_id: str) -> str:
    digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:12]
    return f"CID.{digest}"


def mine_category_patterns(
    train_records: list[NormalizedComplaint],
    taxonomy: TaxonomyFile,
    top_k: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[NormalizedComplaint]] = defaultdict(list)
    for record in train_records:
        grouped[record.normalized_category].append(record)

    patterns: list[dict[str, Any]] = []
    for category in taxonomy.categories:
        records = sorted(grouped.get(category.category_id, []), key=lambda item: item.complaint_id)
        texts = [record.complaint_text_clean for record in records]
        unigram_counts, bigram_counts = _collect_term_counts(texts=texts)

        symptom_terms = _select_terms(
            unigrams=unigram_counts,
            bigrams=bigram_counts,
            stems=SYMPTOM_STEMS,
            fallback_terms=category.keywords_tr,
            top_k=top_k,
        )
        context_terms = _select_terms(
            unigrams=unigram_counts,
            bigrams=bigram_counts,
            stems=CONTEXT_STEMS,
            fallback_terms=category.keywords_tr,
            top_k=top_k,
        )
        trigger_terms = _select_terms(
            unigrams=unigram_counts,
            bigrams=bigram_counts,
            stems=TRIGGER_STEMS,
            fallback_terms=category.example_phrases_tr,
            top_k=top_k,
        )

        example_ids = [_masked_complaint_id(record.complaint_id) for record in records[:5]]
        avg_len = round(mean([len(record.complaint_text_clean) for record in records]), 2) if records else 0.0

        patterns.append(
            {
                "category_id": category.category_id,
                "top_symptoms": symptom_terms,
                "top_context_terms": context_terms,
                "top_trigger_terms": trigger_terms,
                "example_complaint_ids": example_ids,
                "notes": (
                    f"train_records={len(records)}; avg_text_len={avg_len}; "
                    f"fallback_used={'yes' if len(symptom_terms) < 3 else 'no'}"
                ),
            }
        )

    return patterns

