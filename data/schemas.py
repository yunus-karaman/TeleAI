from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, field_validator, model_validator


CONTRACT_VERSION = "1.0.0"


class ContractBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_name: StrictStr
    schema_version: StrictStr = CONTRACT_VERSION
    schema_revision: StrictInt = 1


class RawComplaint(ContractBase):
    schema_name: Literal["RawComplaint"] = "RawComplaint"

    complaint_id: StrictStr | None = None
    source_complaint_id: StrictStr | None = None
    url: StrictStr | None = None
    brand_name: StrictStr = Field(min_length=1)
    brand_slug: StrictStr = Field(pattern=r"^[a-z0-9-]+$")
    title: StrictStr = Field(min_length=3)
    complaint_text: StrictStr = Field(min_length=20)
    created_at_iso: StrictStr | None = None
    scraped_at_iso: StrictStr | None = None
    normalized_category: StrictStr = Field(min_length=1)
    original_category_label: StrictStr | None = None
    tags: list[StrictStr] = Field(default_factory=list)
    support_count: StrictInt = Field(ge=0)
    is_synthetic: StrictBool
    quality_flags: list[StrictStr] = Field(default_factory=list)
    http_status: StrictInt | None = Field(default=None, ge=100, le=599)
    parse_version: StrictStr | None = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: StrictStr | None) -> StrictStr | None:
        if value is None:
            return value
        if not (value.startswith("http://") or value.startswith("https://")):
            raise ValueError("url must start with http:// or https://")
        return value

    @field_validator("created_at_iso", "scraped_at_iso")
    @classmethod
    def validate_iso_datetime(cls, value: StrictStr | None) -> StrictStr | None:
        if value is None:
            return value
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value


class CleanComplaint(ContractBase):
    schema_name: Literal["CleanComplaint"] = "CleanComplaint"

    complaint_id: StrictStr = Field(min_length=1)
    brand_name: StrictStr | None = None
    brand_slug: StrictStr | None = Field(default=None, pattern=r"^[a-z0-9-]+$")
    created_at_iso: StrictStr | None = None
    normalized_category: StrictStr | None = None
    original_category_label: StrictStr | None = None
    title_clean: StrictStr | None = None
    complaint_text_clean: StrictStr = Field(min_length=20)
    tags: list[StrictStr] = Field(default_factory=list)
    support_count: StrictInt = Field(default=0, ge=0)
    quality_flags: list[StrictStr] = Field(default_factory=list)
    preprocess_version: StrictStr = Field(min_length=1)
    preprocess_timestamp_iso: StrictStr
    source_hash_sha256: StrictStr = Field(pattern=r"^[a-f0-9]{64}$")
    duplicate_cluster_id: StrictStr | None = None
    is_duplicate_of: StrictStr | None = None

    @field_validator("created_at_iso", "preprocess_timestamp_iso")
    @classmethod
    def validate_iso_datetime(cls, value: StrictStr | None) -> StrictStr | None:
        if value is None:
            return value
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value

    @model_validator(mode="after")
    def validate_duplicate_metadata(self) -> "CleanComplaint":
        if self.is_duplicate_of and not self.duplicate_cluster_id:
            raise ValueError("duplicate_cluster_id is required when is_duplicate_of is set")
        return self


class NormalizedComplaint(ContractBase):
    schema_name: Literal["NormalizedComplaint"] = "NormalizedComplaint"

    complaint_id: StrictStr = Field(min_length=1)
    brand_name: StrictStr | None = None
    brand_slug: StrictStr | None = Field(default=None, pattern=r"^[a-z0-9-]+$")
    created_at_iso: StrictStr | None = None
    title_clean: StrictStr | None = None
    complaint_text_clean: StrictStr = Field(min_length=20)
    normalized_category: StrictStr = Field(min_length=1)
    confidence_score: StrictFloat = Field(ge=0.0, le=1.0)
    assignment_reason: StrictStr = Field(min_length=3)
    needs_review: StrictBool
    source_category: StrictStr | None = None
    quality_flags: list[StrictStr] = Field(default_factory=list)
    duplicate_cluster_id: StrictStr | None = None
    is_duplicate_of: StrictStr | None = None
    taxonomy_version: StrictStr = Field(min_length=1)
    source_hash_sha256: StrictStr = Field(pattern=r"^[a-f0-9]{64}$")

    @field_validator("created_at_iso")
    @classmethod
    def validate_created_at_iso(cls, value: StrictStr | None) -> StrictStr | None:
        if value is None:
            return value
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value


class SolutionStep(ContractBase):
    schema_name: Literal["SolutionStep"] = "SolutionStep"

    step_id: StrictStr = Field(pattern=r"^STEP\.[A-Z0-9_]+\.\d{3}$")
    category_id: StrictStr = Field(min_length=1)
    level: Literal["L1", "L2", "L3"]
    title_tr: StrictStr = Field(min_length=5, max_length=120)
    instructions_tr: list[StrictStr] = Field(min_length=3, max_length=6)
    required_inputs: list[StrictStr] = Field(default_factory=list)
    success_check: StrictStr = Field(min_length=8, max_length=280)
    stop_conditions: list[StrictStr] = Field(min_length=1, max_length=4)
    escalation_unit: Literal[
        "BILLING_SUPPORT",
        "TECH_SUPPORT_HOME",
        "TECH_SUPPORT_MOBILE",
        "NETWORK_NOC",
        "STORE",
        "PORTING_TEAM",
        "DIGITAL_SUPPORT",
        "GENERAL_SUPPORT",
    ]
    risk_level: Literal["low", "medium", "high"]
    tags: list[StrictStr] = Field(default_factory=list)
    version: StrictStr = Field(min_length=1)

    @field_validator("instructions_tr")
    @classmethod
    def validate_instructions_length(cls, values: list[StrictStr]) -> list[StrictStr]:
        for value in values:
            if len(value.strip()) < 10:
                raise ValueError("Each instruction bullet must be at least 10 chars.")
        return values


class KBParagraph(ContractBase):
    schema_name: Literal["KBParagraph"] = "KBParagraph"

    doc_id: StrictStr = Field(pattern=r"^KB\.[A-Z0-9_]+\.\d{3}$")
    paragraph_id: StrictStr = Field(pattern=r"^KB\.[A-Z0-9_]+\.\d{3}#P\d+$")
    text_tr: StrictStr = Field(min_length=20, max_length=600)
    applies_to_step_ids: list[StrictStr] = Field(min_length=1)
    source_type: Literal["internal_best_practice"]
    confidence: StrictFloat = Field(ge=0.6, le=0.95)
    version: StrictStr = Field(min_length=1)


GraphAttribute = StrictStr | StrictInt | StrictFloat | StrictBool


class GraphNode(ContractBase):
    schema_name: Literal["GraphNode"] = "GraphNode"

    node_id: StrictStr = Field(min_length=1)
    node_type: Literal[
        "complaint",
        "customer_intent",
        "issue",
        "product",
        "brand",
        "time_bucket",
        "category",
        "kb_paragraph",
        "solution_step",
        "policy",
    ]
    label: StrictStr = Field(min_length=1)
    attributes: dict[StrictStr, GraphAttribute] = Field(default_factory=dict)
    source_ids: list[StrictStr] = Field(default_factory=list)
    confidence: StrictFloat = Field(ge=0.0, le=1.0)


class GraphEdge(ContractBase):
    schema_name: Literal["GraphEdge"] = "GraphEdge"

    edge_id: StrictStr = Field(min_length=1)
    source_node_id: StrictStr = Field(min_length=1)
    target_node_id: StrictStr = Field(min_length=1)
    relation_type: Literal[
        "MENTIONS",
        "HAS_CATEGORY",
        "CLASSIFIED_AS",
        "RECOMMENDS_STEP",
        "HAS_BRAND",
        "HAS_TIME_BUCKET",
        "SUPPORTED_BY",
        "RESOLVED_BY",
        "SIMILAR_TO",
        "CONTRADICTS",
        "ESCALATES_TO",
        "DERIVED_FROM",
    ]
    weight: StrictFloat = Field(ge=0.0, le=1.0)
    evidence_ids: list[StrictStr] = Field(default_factory=list)
    bidirectional: StrictBool = False


class TrainingExample(ContractBase):
    schema_name: Literal["TrainingExample"] = "TrainingExample"

    example_id: StrictStr = Field(min_length=1)
    complaint_id: StrictStr = Field(min_length=1)
    split: Literal["train", "val", "test"]
    instruction: StrictStr = Field(min_length=10)
    input_text: StrictStr = Field(min_length=20)
    target_text: StrictStr = Field(min_length=20)
    evidence_ids: list[StrictStr] = Field(min_length=1)
    difficulty: Literal["easy", "medium", "hard"]
    quality_score: StrictFloat = Field(ge=0.0, le=1.0)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    role: Literal["system", "user", "assistant"]
    content: StrictStr = Field(min_length=1)


class Task2SFTExample(ContractBase):
    schema_name: Literal["Task2SFTExample"] = "Task2SFTExample"

    example_id: StrictStr = Field(min_length=1)
    complaint_id: StrictStr = Field(min_length=1)
    split: Literal["train", "val"]
    system_prompt: StrictStr = Field(min_length=20)
    user_message: StrictStr = Field(min_length=20)
    assistant_message: StrictStr = Field(min_length=80)
    normalized_category: StrictStr = Field(min_length=1)
    category_confidence: StrictFloat = Field(ge=0.0, le=1.0)
    allowed_step_ids: list[StrictStr] = Field(min_length=1, max_length=8)
    allowed_evidence_ids: list[StrictStr] = Field(min_length=1, max_length=24)
    messages: list[ChatMessage] = Field(min_length=3, max_length=3)
    source_hash_sha256: StrictStr = Field(pattern=r"^[a-f0-9]{64}$")
    dataset_version: StrictStr = Field(min_length=1)

    @model_validator(mode="after")
    def validate_role_order(self) -> "Task2SFTExample":
        roles = [message.role for message in self.messages]
        if roles != ["system", "user", "assistant"]:
            raise ValueError("messages roles must be exactly [system, user, assistant]")
        if self.messages[0].content != self.system_prompt:
            raise ValueError("messages[0] must match system_prompt")
        if self.messages[1].content != self.user_message:
            raise ValueError("messages[1] must match user_message")
        if self.messages[2].content != self.assistant_message:
            raise ValueError("messages[2] must match assistant_message")
        return self


class Task1IntentExample(ContractBase):
    schema_name: Literal["Task1IntentExample"] = "Task1IntentExample"

    example_id: StrictStr = Field(min_length=1)
    complaint_id: StrictStr = Field(min_length=1)
    split: Literal["train", "val"]
    instruction: StrictStr = Field(min_length=15)
    user_message: StrictStr = Field(min_length=20)
    assistant_message: StrictStr = Field(min_length=1)
    label_category_id: StrictStr = Field(min_length=1)
    source_hash_sha256: StrictStr = Field(pattern=r"^[a-f0-9]{64}$")
    dataset_version: StrictStr = Field(min_length=1)


class LiveAgentHandoffPacket(ContractBase):
    schema_name: Literal["LiveAgentHandoffPacket"] = "LiveAgentHandoffPacket"

    session_id: StrictStr = Field(min_length=1)
    normalized_category: StrictStr = Field(min_length=1)
    category_confidence: StrictFloat = Field(ge=0.0, le=1.0)
    complaint_summary_masked: StrictStr = Field(min_length=10, max_length=1200)
    steps_tried: list[StrictStr] = Field(default_factory=list)
    evidence_ids_used: list[StrictStr] = Field(default_factory=list)
    clarifying_answers: list[StrictStr] = Field(default_factory=list)
    recommended_escalation_unit: Literal[
        "BILLING_SUPPORT",
        "TECH_SUPPORT_HOME",
        "TECH_SUPPORT_MOBILE",
        "NETWORK_NOC",
        "STORE",
        "PORTING_TEAM",
        "DIGITAL_SUPPORT",
        "GENERAL_SUPPORT",
    ]
    escalation_reason: StrictStr = Field(min_length=5, max_length=320)
    attempts: StrictInt = Field(ge=1, le=10)
    generated_at_iso: StrictStr

    @field_validator("generated_at_iso")
    @classmethod
    def validate_generated_at_iso(cls, value: StrictStr) -> StrictStr:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value


class EvidencePack(ContractBase):
    schema_name: Literal["EvidencePack"] = "EvidencePack"

    class TopStepItem(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        step_id: StrictStr = Field(pattern=r"^STEP\.[A-Z0-9_]+\.\d{3}$")
        title_tr: StrictStr = Field(min_length=3)
        level: Literal["L1", "L2", "L3"]
        instructions_tr: list[StrictStr] = Field(min_length=3, max_length=6)
        evidence_ids: list[StrictStr] = Field(min_length=1)
        step_score: StrictFloat = Field(ge=0.0, le=1.0)

    class EvidenceItem(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        paragraph_id: StrictStr = Field(pattern=r"^KB\.[A-Z0-9_]+\.\d{3}#P\d+$")
        text_tr: StrictStr = Field(min_length=20)
        confidence: StrictFloat = Field(ge=0.0, le=1.0)

    class EscalationSuggestion(BaseModel):
        model_config = ConfigDict(extra="forbid", strict=True)
        unit: Literal[
            "BILLING_SUPPORT",
            "TECH_SUPPORT_HOME",
            "TECH_SUPPORT_MOBILE",
            "NETWORK_NOC",
            "STORE",
            "PORTING_TEAM",
            "DIGITAL_SUPPORT",
            "GENERAL_SUPPORT",
        ]
        reason: StrictStr = Field(min_length=5, max_length=240)
        threshold_signals: list[StrictStr] = Field(default_factory=list)

    request_id: StrictStr = Field(min_length=1)
    normalized_category: StrictStr = Field(min_length=1)
    category_confidence: StrictFloat = Field(ge=0.0, le=1.0)
    top_steps: list[TopStepItem] = Field(min_length=1, max_length=8)
    evidence: list[EvidenceItem] = Field(min_length=1)
    escalation_suggestion: EscalationSuggestion
    retrieval_debug: dict[StrictStr, Any] | None = None

    @model_validator(mode="after")
    def validate_top_steps_have_evidence(self) -> "EvidencePack":
        evidence_ids = {item.paragraph_id for item in self.evidence}
        for step in self.top_steps:
            if not step.evidence_ids:
                raise ValueError("Each top step must include at least one evidence_id")
            for evidence_id in step.evidence_ids:
                if evidence_id not in evidence_ids:
                    raise ValueError(f"Step references unknown evidence_id: {evidence_id}")
        return self


class EvaluationReport(ContractBase):
    schema_name: Literal["EvaluationReport"] = "EvaluationReport"

    run_id: StrictStr = Field(min_length=1)
    mode: Literal["SMOKE", "FULL"]
    dataset_size: StrictInt = Field(ge=0)
    valid_records: StrictInt = Field(ge=0)
    quarantined_records: StrictInt = Field(ge=0)
    hallucination_rate: StrictFloat = Field(ge=0.0, le=1.0)
    evidence_coverage: StrictFloat = Field(ge=0.0, le=1.0)
    escalation_rate: StrictFloat = Field(ge=0.0, le=1.0)
    latency_p95_ms: StrictFloat = Field(ge=0.0)
    pass_fail: Literal["PASS", "FAIL"]
    notes: list[StrictStr] = Field(default_factory=list)
    metrics: dict[StrictStr, StrictFloat] = Field(default_factory=dict)
    generated_at_iso: StrictStr

    @field_validator("generated_at_iso")
    @classmethod
    def validate_timestamp(cls, value: StrictStr) -> StrictStr:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value

    @model_validator(mode="after")
    def validate_counts(self) -> "EvaluationReport":
        if self.valid_records + self.quarantined_records > self.dataset_size:
            raise ValueError("valid_records + quarantined_records cannot exceed dataset_size")
        return self


SchemaObject = (
    RawComplaint
    | CleanComplaint
    | NormalizedComplaint
    | SolutionStep
    | KBParagraph
    | GraphNode
    | GraphEdge
    | TrainingExample
    | Task2SFTExample
    | Task1IntentExample
    | LiveAgentHandoffPacket
    | EvidencePack
    | EvaluationReport
)


def serialize_schema_object(obj: SchemaObject) -> dict[str, Any]:
    """Serialize schema objects consistently for JSONL artifacts."""
    return obj.model_dump(mode="json")
