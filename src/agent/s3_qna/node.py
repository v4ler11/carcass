"""Stage 3: answer every planned question with verified citations."""

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from agent.citations import (
    run_agent_with_verified_citations,
    validate_quick_citations,
)
from agent.prompts.quick_citation import CITATION_PROMPT
from agent.prompts.s3_qna import render_question_input, render_system_prompt
from agent.s1_overlay_resolution.models import VerifiedParcelContext
from agent.s2_fan_out_questions.models import (
    AtomicQuestion,
    ExpectedAnswerType,
    QuestionPlan,
)
from agent.s3_qna.models import (
    AnswerStatus,
    QuestionAnswer,
    VerifiedQuestionAnswer,
)
from agent.tools import ALL_TOOLS
from chat.clients import Client
from chat.tools import ToolContext


def _validate_answer(answer: QuestionAnswer, question: AtomicQuestion) -> list[str]:
    problems: list[str] = []
    if answer.status == AnswerStatus.not_found:
        if answer.citations:
            problems.append("not_found answers must not include citations")
        if answer.answer_value is not None:
            problems.append("not_found answers must have a null answer_value")
    elif not answer.citations:
        problems.append("non-not_found answers must include at least one citation")
    if answer.contradicts_planner_rationale and not answer.discrepancy_note:
        problems.append(
            "contradicts_planner_rationale=true requires a discrepancy_note"
        )

    value = answer.answer_value
    expected = question.expected_answer_type
    if value is not None:
        valid = (
            (expected == ExpectedAnswerType.boolean and isinstance(value, bool))
            or (
                expected == ExpectedAnswerType.numeric
                and isinstance(value, int | float)
                and not isinstance(value, bool)
            )
            or (
                expected in {ExpectedAnswerType.categorical, ExpectedAnswerType.text}
                and isinstance(value, str)
            )
            or (
                expected == ExpectedAnswerType.citation_list
                and isinstance(value, list)
                and all(isinstance(item, str) for item in value)
            )
        )
        if not valid:
            problems.append(
                f"answer_value does not match expected_answer_type={expected.value}"
            )
    return problems


def _validate_dependencies(questions: Sequence[AtomicQuestion]) -> None:
    ids = [question.id for question in questions]
    duplicated = sorted(
        {question_id for question_id in ids if ids.count(question_id) > 1}
    )
    if duplicated:
        raise ValueError("duplicate question ids: " + ", ".join(duplicated))
    known = set(ids)
    unknown = sorted(
        {
            dependency
            for question in questions
            for dependency in question.depends_on
            if dependency not in known
        }
    )
    if unknown:
        raise ValueError("unknown depends_on ids: " + ", ".join(unknown))

    visiting: set[str] = set()
    done: set[str] = set()
    by_id = {question.id: question for question in questions}

    def visit(question_id: str) -> None:
        if question_id in done:
            return
        if question_id in visiting:
            raise ValueError(f"cyclic depends_on involving {question_id!r}")
        visiting.add(question_id)
        for dependency in by_id[question_id].depends_on:
            visit(dependency)
        visiting.remove(question_id)
        done.add(question_id)

    for question in questions:
        visit(question.id)


async def s3_qna_node(
    *,
    client: Client,
    model: str,
    parcel_context: VerifiedParcelContext,
    plan: QuestionPlan,
    max_steps: int = 10,
    max_attempts: int = 2,
    max_citation_attempts: int = 3,
    max_concurrency: int = 8,
    user_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    **params: Any,
) -> dict[str, VerifiedQuestionAnswer]:
    """Answer all questions in parallel, respecting dependency ordering."""
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    _validate_dependencies(plan.questions)

    system_prompt = (
        render_system_prompt(
            confirmed_parcel_context=parcel_context.model_dump_json(
                indent=2, by_alias=True
            )
        )
        + "\n\n"
        + CITATION_PROMPT
    )
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks: dict[str, asyncio.Task[VerifiedQuestionAnswer]] = {}

    async def run_one(question: AtomicQuestion) -> VerifiedQuestionAnswer:
        dependency_context = []
        for dependency in question.depends_on:
            answer = await tasks[dependency]
            dependency_context.append(
                {
                    "question_id": dependency,
                    "answer_text": answer.answer_text,
                    "status": answer.status.value,
                }
            )
        async with semaphore:
            return await _answer_question(
                client=client,
                model=model,
                system_prompt=system_prompt,
                question=question,
                dependency_context=dependency_context,
                max_steps=max_steps,
                max_attempts=max_attempts,
                max_citation_attempts=max_citation_attempts,
                user_id=user_id,
                metadata=metadata,
                extra_body=extra_body,
                **params,
            )

    for question in plan.questions:
        tasks[question.id] = asyncio.create_task(run_one(question))

    answers = await asyncio.gather(*(tasks[question.id] for question in plan.questions))
    return {question.id: answer for question, answer in zip(plan.questions, answers)}


async def _answer_question(
    *,
    client: Client,
    model: str,
    system_prompt: str,
    question: AtomicQuestion,
    dependency_context: Sequence[Mapping[str, str]],
    max_steps: int,
    max_attempts: int,
    max_citation_attempts: int,
    user_id: str | None,
    metadata: Mapping[str, Any] | None,
    extra_body: Mapping[str, Any] | None,
    **params: Any,
) -> VerifiedQuestionAnswer:
    stage_metadata: dict[str, Any] = {
        "stage": "s3_qna",
        "step": "answer_question",
        "step_n": 1,
        "question_id": question.id,
        "category": question.category,
        **(metadata or {}),
    }

    (
        parsed,
        verified_citations,
        _response,
        _usage,
        _turn,
    ) = await run_agent_with_verified_citations(
        ToolContext(),
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": render_question_input(question, dependency_context),
            },
        ],
        response_model=QuestionAnswer,
        tools=ALL_TOOLS,
        guardrails=[
            validate_quick_citations,
            lambda answer: _validate_answer(answer, question),
        ],
        max_steps=max_steps,
        max_attempts=max_attempts,
        max_citation_attempts=max_citation_attempts,
        user_id=user_id,
        metadata=stage_metadata,
        extra_body=extra_body,
        **params,
    )

    return VerifiedQuestionAnswer(
        **parsed.model_dump(exclude={"citations"}),
        citations=verified_citations,
    )
