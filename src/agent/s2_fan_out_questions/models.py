"""Response model for stage 2 atomic question planning."""

from enum import StrEnum

from pydantic import BaseModel, Field


class ExpectedAnswerType(StrEnum):
    categorical = "categorical"
    numeric = "numeric"
    boolean = "boolean"
    text = "text"
    citation_list = "citation_list"


class AtomicQuestion(BaseModel):
    """One bounded question for a downstream retrieval agent."""

    id: str = Field(
        ...,
        pattern=r"^Q\d{2,}$",
        description="Zero-padded question identifier, unique within this plan (Q01, Q02, ...).",
    )
    category: str = Field(
        ...,
        min_length=1,
        description=(
            "Seed-checklist category this question covers, or a new slug for a "
            "program- or parcel-specific category not present in the seed."
        ),
    )
    question_text: str = Field(
        ...,
        min_length=1,
        description=(
            "Precise, atomic question answerable from a bounded piece of "
            "regulation text, an OAP passage, or a metadata field."
        ),
    )
    expected_answer_type: ExpectedAnswerType = Field(
        ...,
        description="Shape of the answer the worker agent should return.",
    )
    likely_source_documents: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Most likely document section(s), table(s), or OAP chapter(s) for "
            "the answer, so retrieval can be scoped rather than corpus-wide."
        ),
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description=(
            "IDs of questions that must be answered before this one; empty "
            "when the question is independently answerable."
        ),
    )
    rationale: str = Field(
        ...,
        min_length=1,
        description="Why this question is necessary for the final buildability report.",
    )


class CarryForwardLimitation(BaseModel):
    """An unresolved S1 flag that no text-retrieval question can resolve."""

    flag: str = Field(
        ...,
        min_length=1,
        description="The unresolved flag carried forward verbatim from stage 1.",
    )
    reason: str = Field(
        ...,
        min_length=1,
        description=(
            "Why this flag is a graphic-resolution, geometry, or other "
            "non-text limitation that a retrieval agent cannot resolve."
        ),
    )


class QuestionPlan(BaseModel):
    """The complete downstream question plan for one parcel and program."""

    questions: list[AtomicQuestion] = Field(
        ...,
        min_length=1,
        description=(
            "Complete non-redundant question list covering every seed "
            "category and any additional parcel/program-specific issues."
        ),
    )
    carry_forward_limitations: list[CarryForwardLimitation] = Field(
        default_factory=list,
        description=(
            "Unresolved S1 flags that cannot be answered from text; these are "
            "explicit limitations, not questions."
        ),
    )
