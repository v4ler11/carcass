"""Models for stage 4 final synthesis and its merged answer store."""

from pydantic import BaseModel, Field

from agent.citations import QuickCitation
from agent.s2_fan_out_questions.models import (
    AtomicQuestion,
    CarryForwardLimitation,
)
from agent.s3_qna.models import VerifiedQuestionAnswer


class MergedQuestionAnswer(BaseModel):
    """One Stage 2 question paired with its verified Stage 3 answer."""

    question: AtomicQuestion = Field(
        ...,
        description="The original atomic question from the Stage 2 plan.",
    )
    answer: VerifiedQuestionAnswer = Field(
        ...,
        description="The independently sourced, cited Stage 3 answer.",
    )


class AnswerStore(BaseModel):
    """All Stage 2 questions merged with their Stage 3 answers."""

    answers: list[MergedQuestionAnswer] = Field(
        ...,
        min_length=1,
        description=(
            "Every Stage 2 question, in plan order, paired with its Stage 3 "
            "answer. This is the evidentiary base for final synthesis."
        ),
    )
    carry_forward_limitations: list[CarryForwardLimitation] = Field(
        default_factory=list,
        description=(
            "Unresolved Stage 1 flags that Stage 2 marked as not resolvable "
            "by text retrieval; final synthesis must carry them into "
            "assumptions_and_unresolved."
        ),
    )


class SynthesisReport(BaseModel):
    """Final narrative buildability study with citations."""

    text: str = Field(
        ...,
        min_length=1,
        description=(
            "Full final study as one text document. It must contain the "
            "requested sections, explicit admissibility_status values, "
            "source_type tags, and inline citation tags."
        ),
    )
    citations: list[QuickCitation] = Field(
        ...,
        min_length=1,
        description=(
            "One entry per inline citation tag in text, using the citation "
            "pipeline's claim/c_id/verification_instructions shape."
        ),
    )
