from __future__ import annotations

import pytest
from pydantic import ValidationError

from data.schemas import SolutionStep


def test_solution_step_schema_valid_record() -> None:
    step = SolutionStep(
        step_id="STEP.BILLING_PAYMENTS.001",
        category_id="BILLING_PAYMENTS",
        level="L1",
        title_tr="Fatura ve Odeme: Sorun profilini netlestir",
        instructions_tr=[
            "Gorulen ana belirtiyi tek cumleyle not et.",
            "Sorunun hangi baglamda basladigini kaydet.",
            "Son degisikligi tarih ve saatle birlikte yaz.",
        ],
        required_inputs=["time_window", "error_code", "city_region"],
        success_check="Belirti, zaman ve baglam net bir kayit haline gelir.",
        stop_conditions=["Belirti 24 saatten uzun surerse L2 adimina gec."],
        escalation_unit="BILLING_SUPPORT",
        risk_level="low",
        tags=["profil", "l1"],
        version="solution-steps-v1",
    )
    assert step.step_id == "STEP.BILLING_PAYMENTS.001"


def test_solution_step_schema_rejects_short_instructions() -> None:
    with pytest.raises(ValidationError):
        SolutionStep(
            step_id="STEP.BILLING_PAYMENTS.001",
            category_id="BILLING_PAYMENTS",
            level="L1",
            title_tr="Kisa",
            instructions_tr=["kisa", "kisa", "kisa"],
            required_inputs=["time_window"],
            success_check="Basarisiz",
            stop_conditions=["Durum degismedi."],
            escalation_unit="BILLING_SUPPORT",
            risk_level="low",
            tags=[],
            version="solution-steps-v1",
        )

