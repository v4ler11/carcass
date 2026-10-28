"""Response models for stage 3 per-question answering."""

from enum import StrEnum

from pydantic import BaseModel, Field

from agent.citations import QuickCitation, VerifiedQuickCitation

type AnswerValue = bool | float | str | list[str] | None


class AnswerStatus(StrEnum):
    confirmed = "confirmed"
    partial = "partial"
    ambiguous = "ambiguous"
    not_found = "not_found"


class AnswerConfidence(StrEnum):
    high = "high"
    medium = "medium"
    low = "low"


class QuestionAnswer(BaseModel):
    """One cited answer to one atomic question."""

    question_id: str = Field(
        ...,
        pattern=r"^Q\d{2,}$",
        description="Question identifier copied from the stage-2 plan.",
    )
    status: AnswerStatus = Field(
        ...,
        description=(
            "confirmed when the source directly answers the question; partial "
            "or ambiguous when it only partly does; not_found when no "
            "supporting text was found."
        ),
    )
    answer_text: str = Field(
        ...,
        min_length=1,
        description=(
            "Concise natural-language answer. Every factual claim carries an "
            "inline citation tag and a matching citations[] entry."
        ),
    )
    answer_value: AnswerValue = Field(
        default=None,
        description=(
            "Typed value matching the question's expected_answer_type, or null "
            "when the answer is textual or unavailable."
        ),
    )
    confidence: AnswerConfidence = Field(
        ...,
        description="Self-assessed confidence in this answer.",
    )
    contradicts_planner_rationale: bool = Field(
        ...,
        description=(
            "True if the independently sourced answer differs from, cannot "
            "confirm, or contradicts a claim in planner_rationale."
        ),
    )
    discrepancy_note: str | None = Field(
        default=None,
        description=(
            "Explanation of the discrepancy when contradicts_planner_rationale "
            "is true; null otherwise."
        ),
    )
    additional_context: str | None = Field(
        default=None,
        description=(
            "Nuance or condition that materially changes the literal answer, "
            "such as a value that depends on an unstated condition."
        ),
    )
    citations: list[QuickCitation] = Field(
        ...,
        description=(
            "One entry per inline citation tag in answer_text. Empty only when "
            "status is not_found."
        ),
    )


class VerifiedQuestionAnswer(QuestionAnswer):
    """A stage-3 answer whose citations carry verification verdicts."""

    citations: list[VerifiedQuickCitation] = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        ...,
        description=(
            "One entry per inline citation tag in answer_text, each carrying "
            "the claim, verification instructions, and independent verdict."
        ),
    )
