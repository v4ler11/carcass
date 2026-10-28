"""Measure how many corrupted citations the verifier still confirms.

For every stage-3 answer found in the traces, the final citations are verified
again as they are ("real") and after three corruptions: wrong facts in the
claim, a completely different claim, and wrong verification instructions.
One JSON line per answer goes to `out/evals/verify_citations_corruption`.
"""

import argparse
import asyncio
import json
from collections.abc import Iterator
from typing import Any

import core.logger  # noqa: F401 - importing configures loguru
from loguru import logger

from agent.citations import QuickCitation, verify_citations
from agent.s3_qna.models import QuestionAnswer
from agent.tools import ALL_TOOLS
from chat.clients import Client
from chat.structured import parse_model_output_json
from conversations import Conversation, Node, list_conversations
from core import BASE_DIR
from core.config import (
    DEFAULT_CHAT_MODEL,
    OPENROUTER_EXTRA_BODY,
    OPENROUTER_URL,
    TRACES_DIR,
    require_openrouter_api_key,
)
from evals.verify_citations_corruption.corrupt import Mode, corrupt_citations
from tracing import configure, shutdown
from tracing.attributes import Attr

OUT_PATH = (
    BASE_DIR / "out" / "evals" / "verify_citations_corruption" / "results.jsonl"
)
VARIANTS = ("real", *(mode.value for mode in Mode))
_META = Attr.LANGFUSE_METADATA_PREFIX


def _walk(node: Node) -> Iterator[Node]:
    yield node
    for child in node.children:
        yield from _walk(child)


def final_answer(conversation: Conversation) -> QuestionAnswer:
    """The answer the worker ended with, after any citation correction.

    The last turn of an s3 conversation is usually the verifier, so the answer
    is read from the last turn that produced one: `answer_question`, or
    `correct_citations` when rejected citations were removed.
    """
    conversation.populate()
    turns = [
        turn
        for turn in conversation.tree
        if turn.attributes.get(_META + "step") in ("answer_question", "correct_citations")
    ]
    turn = turns[-1].populate()
    last_chat = [node for node in _walk(turn) if node.name == "chat"][-1]
    content = last_chat.attributes[Attr.OUTPUT_MESSAGES][0]["content"]
    return parse_model_output_json(content, QuestionAnswer)


def s3_conversations(
    limit: int, question_ids: list[str] | None
) -> list[Conversation]:
    found = [c for c in list_conversations() if c.metadata.get("stage") == "s3"]
    if question_ids:
        found = [c for c in found if c.metadata.get("question_id") in question_ids]
    return found[:limit]


async def _verify(
    *,
    client: Client,
    model: str,
    answer: QuestionAnswer,
    citations: list[QuickCitation],
    metadata: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not citations:
        return {}
    verdicts = await verify_citations(
        client=client,
        model=model,
        structured_output=answer.model_copy(update={"citations": citations}),
        citation_ids=[c.id for c in citations],
        tools=ALL_TOOLS,
        metadata=metadata,
        extra_body=OPENROUTER_EXTRA_BODY,
    )
    return {
        v.c_id: {"confirmed": v.confirmed, "reasoning": v.reasoning} for v in verdicts
    }


async def evaluate_answer(
    *, client: Client, model: str, conversation: Conversation
) -> dict[str, Any]:
    question_id = conversation.metadata["question_id"]
    answer = final_answer(conversation)
    metadata = {"stage": "eval_verify_citations_corruption", "question_id": question_id}

    variants: dict[str, list[QuickCitation]] = {"real": list(answer.citations)}
    if answer.citations:
        corrupted = await asyncio.gather(
            *(
                corrupt_citations(
                    client=client,
                    model=model,
                    citations=answer.citations,
                    mode=mode,
                    metadata=metadata,
                )
                for mode in Mode
            )
        )
        variants.update({mode.value: cits for mode, cits in zip(Mode, corrupted)})
    else:
        variants.update({mode.value: [] for mode in Mode})

    verdicts = await asyncio.gather(
        *(
            _verify(
                client=client,
                model=model,
                answer=answer,
                citations=variants[name],
                metadata={**metadata, "step": "verify", "variant": name},
            )
            for name in VARIANTS
        )
    )

    record: dict[str, Any] = {
        "question_id": question_id,
        "conversation_id": conversation.conversation_id,
    }
    for name, validated in zip(VARIANTS, verdicts):
        record[name] = {
            "citations": {
                c.id: {
                    "claim": c.claim,
                    "verification_instructions": c.verification_instructions,
                }
                for c in variants[name]
            },
            "validated_citations": validated,
        }
    return record


def summarize(records: list[dict[str, Any]]) -> dict[str, dict[str, int | float | None]]:
    summary: dict[str, dict[str, int | float | None]] = {}
    for name in VARIANTS:
        total = sum(len(r[name]["citations"]) for r in records)
        confirmed = sum(
            v["confirmed"] for r in records for v in r[name]["validated_citations"].values()
        )
        summary[name] = {
            "citations": total,
            "confirmed": confirmed,
            "confirmed_rate": round(confirmed / total, 4) if total else None,
        }
    return summary


def _load_records() -> list[dict[str, Any]]:
    if not OUT_PATH.exists():
        return []
    return [
        json.loads(line)
        for line in OUT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


async def run_eval(
    *, limit: int, question_ids: list[str] | None, concurrency: int
) -> None:
    done = _load_records()
    conversations = [
        c
        for c in s3_conversations(limit, question_ids)
        if c.metadata["question_id"] not in {r["question_id"] for r in done}
    ]
    logger.info(
        "{} answers already in {}, evaluating {} more",
        len(done),
        OUT_PATH.name,
        len(conversations),
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    records: list[dict[str, Any]] = list(done)
    failures: list[str] = []

    configure(TRACES_DIR)
    try:
        with OUT_PATH.open("a", encoding="utf-8") as out:
            async with Client(
                base_url=OPENROUTER_URL,
                api_key=require_openrouter_api_key(),
                extra_body=OPENROUTER_EXTRA_BODY,
            ) as client:

                async def one(conversation: Conversation) -> None:
                    question_id = conversation.metadata["question_id"]
                    async with semaphore:
                        try:
                            record = await evaluate_answer(
                                client=client,
                                model=DEFAULT_CHAT_MODEL,
                                conversation=conversation,
                            )
                        except Exception:  # noqa: BLE001 - one bad answer must not stop the run
                            logger.exception("{} failed", question_id)
                            failures.append(question_id)
                            return
                    records.append(record)
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    logger.info("{} done", question_id)

                await asyncio.gather(*(one(c) for c in conversations))
    finally:
        shutdown()

    print(json.dumps({"answers": len(records), "failed": failures, **summarize(records)}, indent=2))
    print(f"wrote {OUT_PATH}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=64, help="s3 answers to evaluate")
    parser.add_argument("--questions", nargs="+", help="only these question ids, e.g. Q41 Q20")
    parser.add_argument("--concurrency", type=int, default=4, help="answers in flight")
    return parser.parse_args()


def run() -> None:
    args = _parse_args()
    asyncio.run(
        run_eval(
            limit=args.limit, question_ids=args.questions, concurrency=args.concurrency
        )
    )


if __name__ == "__main__":
    run()
