"""Run the agent pipeline stages in order.

Stages are sequential: each stage consumes the prior stage's result. A single
stage can be selected from the CLI, or all registered stages can be run.
"""

import argparse
import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger
from pydantic import BaseModel

import core.logger  # noqa: F401 - importing configures loguru
from agent.paths import (
    S1_RESULTS_PATH,
    S2_RESULTS_PATH,
    S3_RESULTS_DIR,
    S4_ANSWER_STORE_PATH,
    S4_MARKDOWN_PATH,
    S4_RESULTS_PATH,
)
from agent.s1_overlay_resolution import (
    VerifiedParcelContext,
    s1_overlay_resolution_node,
)
from agent.s2_fan_out_questions import QuestionPlan, s2_fan_out_questions_node
from agent.s3_qna import VerifiedQuestionAnswer, s3_qna_node
from agent.s4_synthesis import (
    AnswerStore,
    MergedQuestionAnswer,
    render_markdown,
    s4_synthesis_node,
)
from chat.clients import Client
from core.config import (
    DEFAULT_CHAT_MODEL,
    OPENROUTER_EXTRA_BODY,
    OPENROUTER_URL,
    TRACES_DIR,
    require_openrouter_api_key,
)
from tracing import configure, shutdown

Stage = Callable[..., Awaitable[Any]]


async def run_s1(*, client: Client, model: str) -> VerifiedParcelContext:
    """Run stage 1, persist its result, and print the same JSON."""
    logger.info("running agent node {}", s1_overlay_resolution_node.__name__)
    result, _response, _usage, _turn = await s1_overlay_resolution_node(
        client=client,
        model=model,
        metadata={"kind": "main_pipeline", "stage": "s1"},
    )
    payload = result.model_dump_json(indent=2, by_alias=True)
    S1_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    S1_RESULTS_PATH.write_text(payload + "\n")
    logger.info("finished stage s1; wrote {}", S1_RESULTS_PATH)
    print(payload)
    return result


async def run_s2(*, client: Client, model: str) -> BaseModel:
    """Load the S1 result, plan questions, and persist the S2 result."""
    logger.info("loading stage 1 results from {}", S1_RESULTS_PATH)
    parcel_context = VerifiedParcelContext.model_validate_json(
        S1_RESULTS_PATH.read_text()
    )
    logger.info("running agent node {}", s2_fan_out_questions_node.__name__)
    result, _response, _usage, _turn = await s2_fan_out_questions_node(
        client=client,
        model=model,
        parcel_context=parcel_context,
        metadata={"kind": "main_pipeline", "stage": "s2"},
    )
    payload = result.model_dump_json(indent=2, by_alias=True)
    S2_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    S2_RESULTS_PATH.write_text(payload + "\n")
    logger.info("finished stage s2; wrote {}", S2_RESULTS_PATH)
    return result


async def run_s3(*, client: Client, model: str) -> dict[str, Any]:
    """Load S1 and S2, answer every question, and save one file per question."""
    logger.info("loading stage 1 results from {}", S1_RESULTS_PATH)
    parcel_context = VerifiedParcelContext.model_validate_json(
        S1_RESULTS_PATH.read_text()
    )
    logger.info("loading stage 2 questions from {}", S2_RESULTS_PATH)
    plan = QuestionPlan.model_validate_json(S2_RESULTS_PATH.read_text())
    logger.info("running agent node {}", s3_qna_node.__name__)
    answers = await s3_qna_node(
        client=client,
        model=model,
        parcel_context=parcel_context,
        plan=plan,
        metadata={"kind": "main_pipeline", "stage": "s3"},
    )
    S3_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for question_id, answer in answers.items():
        (S3_RESULTS_DIR / f"{question_id}.json").write_text(
            answer.model_dump_json(indent=2, by_alias=True) + "\n"
        )
    logger.info("finished stage s3; wrote {} answers", len(answers))
    return answers


async def run_s4(*, client: Client, model: str) -> BaseModel:
    """Merge S2/S3 answers, synthesize the report, and export Markdown."""
    logger.info("loading stage 1 results from {}", S1_RESULTS_PATH)
    parcel_context = VerifiedParcelContext.model_validate_json(
        S1_RESULTS_PATH.read_text()
    )
    logger.info("loading stage 2 questions from {}", S2_RESULTS_PATH)
    plan = QuestionPlan.model_validate_json(S2_RESULTS_PATH.read_text())

    merged: list[MergedQuestionAnswer] = []
    missing: list[str] = []
    for question in plan.questions:
        path = S3_RESULTS_DIR / f"{question.id}.json"
        if not path.exists():
            missing.append(question.id)
            continue
        answer = VerifiedQuestionAnswer.model_validate_json(path.read_text())
        merged.append(MergedQuestionAnswer(question=question, answer=answer))
    if missing:
        raise FileNotFoundError(
            "missing stage 3 answers for: " + ", ".join(sorted(missing))
        )

    answer_store = AnswerStore(
        answers=merged,
        carry_forward_limitations=plan.carry_forward_limitations,
    )
    S4_ANSWER_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    S4_ANSWER_STORE_PATH.write_text(
        answer_store.model_dump_json(indent=2, by_alias=True) + "\n"
    )
    logger.info(
        "merged {} questions and answers into {}",
        len(merged),
        S4_ANSWER_STORE_PATH,
    )

    logger.info("running agent node {}", s4_synthesis_node.__name__)
    report, _response, _usage, _turn = await s4_synthesis_node(
        client=client,
        model=model,
        parcel_context=parcel_context,
        answer_store=answer_store,
        metadata={"kind": "main_pipeline", "stage": "s4"},
    )
    S4_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    S4_RESULTS_PATH.write_text(report.model_dump_json(indent=2, by_alias=True) + "\n")
    logger.info("finished stage s4; wrote {}", S4_RESULTS_PATH)

    markdown, _md_response, _md_usage = await render_markdown(
        client=client,
        model=model,
        report=report,
        plan=plan,
    )
    S4_MARKDOWN_PATH.write_text(markdown)
    logger.info("wrote markdown report to {}", S4_MARKDOWN_PATH)
    return report


STAGES: dict[str, Stage] = {
    "s1": run_s1,
    "s2": run_s2,
    "s3": run_s3,
    "s4": run_s4,
}


def _selected_stages(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return tuple(STAGES)
    if stage not in STAGES:
        available = ", ".join((*STAGES, "all"))
        raise ValueError(f"unknown stage {stage!r}; choose from: {available}")
    return (stage,)


async def run_pipeline(
    *, client: Client, model: str, stage: str = "all"
) -> dict[str, Any]:
    """Run `stage` (or every registered stage) in pipeline order."""
    results: dict[str, Any] = {}
    for name in _selected_stages(stage):
        results[name] = await STAGES[name](client=client, model=model)
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        default="all",
        choices=(*STAGES, "all"),
        help="pipeline stage to run (default: all)",
    )
    return parser.parse_args()


async def main(*, stage: str = "all") -> None:
    configure(TRACES_DIR)
    try:
        async with Client(
            base_url=OPENROUTER_URL,
            api_key=require_openrouter_api_key(),
            extra_body=OPENROUTER_EXTRA_BODY,
        ) as client:
            await run_pipeline(client=client, model=DEFAULT_CHAT_MODEL, stage=stage)
    finally:
        shutdown()


def run() -> None:
    asyncio.run(main(stage=_parse_args().stage))


if __name__ == "__main__":
    run()
