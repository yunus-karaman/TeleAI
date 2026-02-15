from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from data.schemas import KBParagraph, SolutionStep
from taxonomy.schema import TaxonomyCategory


class StepKBLink(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    step_id: StrictStr = Field(pattern=r"^STEP\.[A-Z0-9_]+\.\d{3}$")
    evidence_ids: list[StrictStr] = Field(min_length=1)
    rationale: StrictStr = Field(min_length=8, max_length=300)
    version: StrictStr = Field(min_length=1)


ESCALATION_MAP = {
    "BILLING_PAYMENTS": "BILLING_SUPPORT",
    "MOBILE_DATA_SPEED": "TECH_SUPPORT_MOBILE",
    "MOBILE_VOICE_SMS": "TECH_SUPPORT_MOBILE",
    "HOME_INTERNET_FIBER_DSL": "TECH_SUPPORT_HOME",
    "OUTAGE_SERVICE_DOWN": "NETWORK_NOC",
    "COVERAGE_SIGNAL": "NETWORK_NOC",
    "ROAMING_INTERNATIONAL": "TECH_SUPPORT_MOBILE",
    "NUMBER_PORTING_MNP": "PORTING_TEAM",
    "SIM_LINE_ACCOUNT": "STORE",
    "PLANS_PACKAGES_CAMPAIGNS": "BILLING_SUPPORT",
    "CONTRACT_COMMITMENT_CANCELLATION": "BILLING_SUPPORT",
    "INSTALLATION_INFRASTRUCTURE": "TECH_SUPPORT_HOME",
    "MODEM_DEVICE": "TECH_SUPPORT_HOME",
    "DIGITAL_APP_AUTH": "DIGITAL_SUPPORT",
    "CUSTOMER_SUPPORT_PROCESS": "GENERAL_SUPPORT",
    "OTHER": "GENERAL_SUPPORT",
}


def _category_title_tr(category_id: str, taxonomy_map: dict[str, TaxonomyCategory]) -> str:
    category = taxonomy_map.get(category_id)
    if category is None:
        return category_id.replace("_", " ").title()
    return category.title_tr


def _inputs_for_category(category_id: str) -> list[str]:
    if category_id in {"HOME_INTERNET_FIBER_DSL", "MODEM_DEVICE", "INSTALLATION_INFRASTRUCTURE"}:
        return ["modem_led_state", "device_os", "city_region", "time_window"]
    if category_id in {"MOBILE_DATA_SPEED", "MOBILE_VOICE_SMS", "COVERAGE_SIGNAL", "ROAMING_INTERNATIONAL"}:
        return ["device_os", "signal_level", "city_region", "time_window"]
    if category_id in {"BILLING_PAYMENTS", "PLANS_PACKAGES_CAMPAIGNS", "CONTRACT_COMMITMENT_CANCELLATION"}:
        return ["app_version", "plan_package_name", "time_window", "error_code"]
    if category_id in {"NUMBER_PORTING_MNP", "SIM_LINE_ACCOUNT"}:
        return ["device_os", "time_window", "city_region", "error_code"]
    if category_id in {"DIGITAL_APP_AUTH"}:
        return ["device_os", "app_version", "error_code", "time_window"]
    return ["device_os", "time_window", "city_region", "error_code"]


def _build_step_templates(symptom: str, context: str, trigger: str, category_title: str) -> list[dict[str, Any]]:
    return [
        {
            "level": "L1",
            "title_tr": f"{category_title}: Sorun profilini netlestir",
            "instructions_tr": [
                f"Gorulen ana belirtiyi tek cumleyle not et: {symptom}.",
                f"Sorunun hangi baglamda ortaya ciktigini isaretle: {context}.",
                f"Son degisikligi kaydet: {trigger}.",
            ],
            "success_check": "Belirti, zaman ve baglam tek bir kayitta netlesir.",
            "stop_conditions": ["Belirti 24 saatten uzun suruyorsa L2 adimina gec."],
            "risk_level": "low",
            "tags": ["profil", "tespit", "l1"],
        },
        {
            "level": "L1",
            "title_tr": f"{category_title}: Temel servis yenileme",
            "instructions_tr": [
                "Cihazi veya baglanti ortamini guvenli sekilde yeniden baslat.",
                "Ag baglantisini kapat-ac yapip hizmeti tekrar dene.",
                "Ayni islemi 2 dakika arayla bir kez daha tekrarla.",
            ],
            "success_check": "Hizmet surekli hale gelir veya hata sikligi azalir.",
            "stop_conditions": ["Ayni hata iki testte de tekrar ederse L2 adimina gec."],
            "risk_level": "low",
            "tags": ["temel_kontrol", "yenileme", "l1"],
        },
        {
            "level": "L1",
            "title_tr": f"{category_title}: Durum ve kapsam kontrolu",
            "instructions_tr": [
                "Resmi hizmet durum ekranindan genel kesinti bilgisini kontrol et.",
                "Ayni ortamda ikinci bir cihazla kisa dogrulama testi yap.",
                "Gorulen hata kodunu veya uyari mesajini kaydet.",
            ],
            "success_check": "Sorunun yerel mi genel mi oldugu ayrisir.",
            "stop_conditions": ["Genel kesinti yoksa ve sorun devam ederse L2 adimina gec."],
            "risk_level": "low",
            "tags": ["kapsam", "durum", "l1"],
        },
        {
            "level": "L2",
            "title_tr": f"{category_title}: Ileri teknik dogrulama",
            "instructions_tr": [
                "Iki farkli zaman penceresinde ayni testi uygulayip sonucu kaydet.",
                "Baglanti modunu veya uygulama surumunu guncelleyip tekrar dene.",
                "Sorun tetikleyicisi ile sonucun bagini not et.",
            ],
            "success_check": "Tekrarlanabilir hata paterni olusur ve teknik tanim netlesir.",
            "stop_conditions": ["Iki farkli denemede ayni hata devam ederse L3 adimina gec."],
            "risk_level": "medium",
            "tags": ["ileri_test", "tekrar", "l2"],
        },
        {
            "level": "L2",
            "title_tr": f"{category_title}: Kanit paketini hazirla",
            "instructions_tr": [
                "Hata zamani, gorulen mesaj ve test sonucunu tek listede birlestir.",
                "Kullanilan cihaz tipi, isletim sistemi ve genel bolge bilgisini ekle.",
                "Hizmetteki etkisini kisa ve olculebilir ifadeyle yaz.",
            ],
            "success_check": "Destek ekibinin tekrar sormadan kullanabilecegi ozet hazir olur.",
            "stop_conditions": ["Kanit paketi hazir olmasina ragmen sorun surerse L3 adimina gec."],
            "risk_level": "medium",
            "tags": ["kanit", "dokumantasyon", "l2"],
        },
        {
            "level": "L3",
            "title_tr": f"{category_title}: Yapilandirilmis destek aktarimi",
            "instructions_tr": [
                "Destek kaydina yalniz teknik bulgu ve test zamanlarini aktar.",
                "Acil beklentiyi net yaz: hizmetin geri gelmesi, stabilite veya duzeltme.",
                "Takip icin kayit numarasi ve geri donus penceresi iste.",
            ],
            "success_check": "Kayit uygun birime yonlenir ve takip adimi netlesir.",
            "stop_conditions": ["Geri donus penceresi asilirsa ust destek kanalina aktar."],
            "risk_level": "high",
            "tags": ["eskalasyon", "takip", "l3"],
        },
    ]


def generate_solution_steps_for_category(
    *,
    category_id: str,
    category_pattern: dict[str, Any],
    taxonomy_map: dict[str, TaxonomyCategory],
    version: str,
) -> list[SolutionStep]:
    symptoms = category_pattern.get("top_symptoms", [])
    contexts = category_pattern.get("top_context_terms", [])
    triggers = category_pattern.get("top_trigger_terms", [])
    symptom = symptoms[0] if symptoms else "hizmet sorunu"
    context = contexts[0] if contexts else "standart kullanim"
    trigger = triggers[0] if triggers else "son degisiklik"

    category_title = _category_title_tr(category_id=category_id, taxonomy_map=taxonomy_map)
    escalation_unit = ESCALATION_MAP.get(category_id, "GENERAL_SUPPORT")
    required_inputs = _inputs_for_category(category_id=category_id)
    templates = _build_step_templates(
        symptom=symptom,
        context=context,
        trigger=trigger,
        category_title=category_title,
    )

    steps: list[SolutionStep] = []
    for index, template in enumerate(templates, start=1):
        step_id = f"STEP.{category_id}.{index:03d}"
        risk = template["risk_level"]
        if template["level"] == "L3":
            taxonomy_risk = taxonomy_map[category_id].risk_level_default if category_id in taxonomy_map else "medium"
            risk = "high" if taxonomy_risk == "high" else "medium"
        steps.append(
            SolutionStep(
                step_id=step_id,
                category_id=category_id,
                level=template["level"],
                title_tr=template["title_tr"],
                instructions_tr=template["instructions_tr"],
                required_inputs=required_inputs,
                success_check=template["success_check"],
                stop_conditions=template["stop_conditions"],
                escalation_unit=escalation_unit,
                risk_level=risk,
                tags=sorted(set(template["tags"] + [category_id.lower()])),
                version=version,
            )
        )
    return steps


def _kb_confidence(level: str, paragraph_index: int) -> float:
    if level == "L1":
        return 0.82 if paragraph_index == 1 else 0.77
    if level == "L2":
        return 0.85 if paragraph_index == 1 else 0.8
    return 0.74 if paragraph_index == 1 else 0.7


def generate_kb_and_links_for_steps(
    *,
    steps: list[SolutionStep],
    version: str,
) -> tuple[list[KBParagraph], list[StepKBLink]]:
    kb_items: list[KBParagraph] = []
    links: list[StepKBLink] = []

    for step in sorted(steps, key=lambda item: item.step_id):
        step_number = step.step_id.split(".")[-1]
        doc_id = f"KB.{step.category_id}.{step_number}"
        paragraph_ids: list[str] = []

        text_1 = (
            f"Bu adim {step.category_id} kapsaminda sik gorulen kesinti ve performans sorunlarini sistematik sekilde "
            "ayirmak icin kullanilir. Temel hedef, sorunun tekrar kosulunu netlestirerek gereksiz yonlendirmeyi azaltmaktir."
        )
        text_2 = (
            "Kayitli test zamani, hata goruntusu ve genel bolge bilgisi teknik ekibin olayi daha hizli siniflandirmasina "
            "yardim eder. Bu yontem, hassas kimlik verisi toplamadan guvenli bir inceleme akisina imkan verir."
        )

        for paragraph_index, text in enumerate([text_1, text_2], start=1):
            paragraph_id = f"{doc_id}#P{paragraph_index}"
            paragraph_ids.append(paragraph_id)
            kb_items.append(
                KBParagraph(
                    doc_id=doc_id,
                    paragraph_id=paragraph_id,
                    text_tr=text,
                    applies_to_step_ids=[step.step_id],
                    source_type="internal_best_practice",
                    confidence=_kb_confidence(step.level, paragraph_index),
                    version=version,
                )
            )

        links.append(
            StepKBLink(
                step_id=step.step_id,
                evidence_ids=paragraph_ids,
                rationale="Adim uygulanirken teknik tanilama baglamini guclendiren temel uygulama notlari.",
                version=version,
            )
        )

    return kb_items, links

