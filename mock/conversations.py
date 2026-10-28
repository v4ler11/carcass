"""Mock conversations that populate `data/traces` with real model calls.

Run from the repository root::

    uv run python mock/conversations.py

Each scenario drives `run_agent` through five user turns (so the transcript
ends 10-20 messages deep, tool round trips included) and then feeds the
transcript to `complete_structured`. Turns chain through `parent=`, so the
spans form one conversation instead of a pile of unrelated requests.

Endpoint, model and `extra_body` mirror `core.server`; the script calls the
bindings directly instead of going through the HTTP server, so no port has to
be listening. After every scenario it flushes the tracer, reads the spans back
through `tracing.read_spans` (which also resolves the out-of-line content
blobs) and prints the resulting span tree, then verifies the invariants a
reader of the traces relies on.
"""

import asyncio
import json
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from chat.agent import run_agent
from chat.clients import Client
from chat.structured import complete_structured
from chat.tools.base import Tool
from chat.tools.context import ToolContext
from chat.usage import ChatUsage
from chat.utils import tool_message
from core.config import (
    MOCK_CHAT_MODEL,
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_EXTRA_BODY,
    OPENROUTER_URL,
    TRACES_DIR,
    openrouter_api_key,
)
from tracing import Turn, configure, read_spans, shutdown
from tracing.attributes import Attr

# Mirrors the module-level `app` in core.server exactly.
MODEL = MOCK_CHAT_MODEL

USER_ID = "mock-user"
TURNS_PER_SCENARIO = 5
DEPTH_MIN, DEPTH_MAX = 10, 20


class MockTool(Tool):
    """A deterministic local tool: the model chooses, the script answers."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[[dict[str, Any]], Any],
    ) -> None:
        self._name = name
        self._description = description
        self._parameters = parameters
        self._handler = handler

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, ctx, tool_call, args) -> list[ChatCompletionMessageParam]:
        answer = self._handler(args)
        return [tool_message(tool_call.id, json.dumps(answer, ensure_ascii=False))]

    def into_chat_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self._name,
                "description": self._description,
                "parameters": self._parameters,
            },
        }


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


# --- Lisbon trip -------------------------------------------------------------

_FORECAST = {
    "april": {"low_c": 12, "high_c": 19, "conditions": "passing showers"},
    "may": {"low_c": 14, "high_c": 22, "conditions": "sunny with a breeze"},
    "june": {"low_c": 17, "high_c": 26, "conditions": "dry and warm"},
}


def forecast(args: dict[str, Any]) -> Any:
    city = str(args.get("city", "Lisbon"))
    month = str(args.get("month", "may")).lower()
    data = _FORECAST.get(month)
    if data is None:
        return {"city": city, "month": month, "error": "no forecast for that month"}
    return {"city": city, "month": month, **data}


_RATES_TO_USD = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "JPY": 0.0067}


def currency(args: dict[str, Any]) -> Any:
    amount = float(args.get("amount", 0))
    source = str(args.get("source", "USD")).upper()
    target = str(args.get("target", "EUR")).upper()
    if source not in _RATES_TO_USD or target not in _RATES_TO_USD:
        return {"error": f"unsupported pair {source}->{target}"}
    converted = amount * _RATES_TO_USD[source] / _RATES_TO_USD[target]
    return {
        "amount": amount,
        "source": source,
        "target": target,
        "converted": round(converted, 2),
    }


# --- Coffee-gear support -----------------------------------------------------

_ORDERS = {
    "A-1042": {
        "item": "espresso tamper",
        "status": "delivered",
        "total_usd": 38.0,
        "delivered_on": "2026-10-11",
    },
    "B-7781": {
        "item": "burr grinder",
        "status": "in_transit",
        "total_usd": 129.0,
        "delivered_on": None,
    },
}


def lookup_order(args: dict[str, Any]) -> Any:
    order_id = str(args.get("order_id", "")).upper()
    order = _ORDERS.get(order_id)
    if order is None:
        return {"order_id": order_id, "error": "order not found"}
    return {"order_id": order_id, **order}


def shipping(args: dict[str, Any]) -> Any:
    destination = str(args.get("destination", "Portland"))
    weight = float(args.get("weight_kg", 1.0))
    return {
        "destination": destination,
        "weight_kg": weight,
        "options": [
            {"carrier": "Ground", "days": 4, "cost_usd": round(4.5 + 2.2 * weight, 2)},
            {
                "carrier": "Two-day",
                "days": 2,
                "cost_usd": round(11.0 + 3.1 * weight, 2),
            },
        ],
    }


# --- Checkout latency retro --------------------------------------------------

_METRICS = {
    "checkout": [280, 295, 300, 312, 340, 365, 410],
    "search": [120, 118, 125, 121, 119, 123, 122],
}
_OWNERS = {"checkout": "payments-platform", "search": "discovery"}


def metric(args: dict[str, Any]) -> Any:
    service = str(args.get("service", "checkout")).lower()
    days = int(args.get("days", 7))
    series = _METRICS.get(service)
    if series is None:
        return {"service": service, "error": "unknown service"}
    window = series[-days:] if days > 0 else series
    return {"service": service, "unit": "ms", "p95": window, "max": max(window)}


def owner(args: dict[str, Any]) -> Any:
    service = str(args.get("service", "checkout")).lower()
    on_call = _OWNERS.get(service)
    if on_call is None:
        return {"service": service, "error": "no owner on call"}
    return {"service": service, "owner": on_call}


# --- Scenarios ---------------------------------------------------------------


class TravelBrief(BaseModel):
    destination: str
    days: int
    estimated_total_usd: float
    packing_note: str
    risks: list[str]


class SupportCase(BaseModel):
    order_id: str
    recommendation: str
    reason: str
    refund_amount_usd: float
    customer_message: str


class RetroSummary(BaseModel):
    service: str
    likely_cause: str
    actions: list[str]
    confidence: float


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    tools: tuple[Tool, ...]
    turns: tuple[str, ...]
    extraction_prompt: str
    response_model: type[BaseModel]


TRAVEL = Scenario(
    key="lisbon-trip",
    title="Four days in Lisbon",
    tools=(
        MockTool(
            "get_forecast",
            "Weather outlook for a city and month.",
            _object(
                {"city": {"type": "string"}, "month": {"type": "string"}},
                ["city", "month"],
            ),
            forecast,
        ),
        MockTool(
            "convert_currency",
            "Convert an amount between currencies at fixed mock rates.",
            _object(
                {
                    "amount": {"type": "number"},
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                },
                ["amount", "source", "target"],
            ),
            currency,
        ),
    ),
    turns=(
        "I'm planning four days in Lisbon in May. Use get_forecast before you tell me what to pack.",
        (
            "A pastel de nata and coffee tour costs 35 EUR per person for four people. "
            "Use convert_currency to give me the total in USD."
        ),
        "The hotel is 620 EUR for four nights. Convert that to USD as well.",
        "My round-trip flights are 410 GBP. Use the converter to tell me what that is in EUR.",
        "Put it all together: a short budget summary and the single biggest risk for the trip.",
    ),
    extraction_prompt=(
        "Convert the conversation into the TravelBrief schema: destination, days, "
        "estimated total cost in USD, one packing note, and the risks."
    ),
    response_model=TravelBrief,
)

SUPPORT = Scenario(
    key="coffee-support",
    title="Missing espresso tamper",
    tools=(
        MockTool(
            "lookup_order",
            "Look up an order by id.",
            _object({"order_id": {"type": "string"}}, ["order_id"]),
            lookup_order,
        ),
        MockTool(
            "shipping_options",
            "Shipping options for a parcel of a given weight and destination.",
            _object(
                {"destination": {"type": "string"}, "weight_kg": {"type": "number"}},
                ["destination", "weight_kg"],
            ),
            shipping,
        ),
    ),
    turns=(
        "A customer says order A-1042 never arrived. Use lookup_order to check its status.",
        "They would rather have the burr grinder from order B-7781 instead. Use lookup_order on that order too.",
        "Get shipping_options for a 2.4 kg replacement shipped to Portland.",
        "Get shipping_options for the same parcel shipped to Honolulu instead.",
        "Write the customer a resolution: replace or refund, and say why.",
    ),
    extraction_prompt=(
        "Convert the conversation into the SupportCase schema: order id, the "
        "recommendation, the reason, any refund amount in USD, and a short "
        "customer-facing message."
    ),
    response_model=SupportCase,
)

RETRO = Scenario(
    key="checkout-retro",
    title="Checkout p95 regression",
    tools=(
        MockTool(
            "query_metric",
            "Daily p95 latency series for a service.",
            _object(
                {"service": {"type": "string"}, "days": {"type": "integer"}},
                ["service", "days"],
            ),
            metric,
        ),
        MockTool(
            "find_owner",
            "The team currently on call for a service.",
            _object({"service": {"type": "string"}}, ["service"]),
            owner,
        ),
    ),
    turns=(
        "Checkout p95 has been climbing. Use query_metric for the last 7 days.",
        "Pull the last 3 days for search as a comparison, with query_metric.",
        "Who owns checkout? Use find_owner.",
        "Who owns search?",
        "Summarize the likely cause and two concrete action items.",
    ),
    extraction_prompt=(
        "Convert the conversation into the RetroSummary schema: the service, the "
        "likely cause, two actions, and your confidence from 0 to 1."
    ),
    response_model=RetroSummary,
)

SCENARIOS = (TRAVEL, SUPPORT, RETRO)


# --- Running -----------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    scenario: Scenario
    conversation_id: str
    transcript: list[ChatCompletionMessageParam]
    usage: ChatUsage


def _tool_rounds(transcript: list[ChatCompletionMessageParam]) -> int:
    return sum(
        1
        for message in transcript
        if message["role"] == "assistant" and message.get("tool_calls")
    )


async def run_scenario(client: Client, scenario: Scenario) -> Run:
    conversation_id = uuid4().hex
    transcript: list[ChatCompletionMessageParam] = []
    usage = ChatUsage(model=MODEL)
    turn: Turn | None = None

    for index, prompt in enumerate(scenario.turns, start=1):
        transcript.append({"role": "user", "content": prompt})
        _response, turn_usage, transcript, turn = await run_agent(
            ToolContext(),
            client=client,
            model=MODEL,
            messages=transcript,
            tools=scenario.tools,
            max_steps=6,
            conversation_id=conversation_id,
            parent=turn,
            user_id=USER_ID,
            metadata={"scenario": scenario.key, "turn": index},
            extra_body=OPENROUTER_EXTRA_BODY,
        )
        usage.consume(turn_usage)

    parsed, _response, structured_usage, _turn = await complete_structured(
        client=client,
        model=MODEL,
        messages=[
            *transcript,
            {"role": "user", "content": scenario.extraction_prompt},
        ],
        response_model=scenario.response_model,
        conversation_id=conversation_id,
        parent=turn,
        user_id=USER_ID,
        metadata={"scenario": scenario.key, "step": "extract"},
        extra_body=OPENROUTER_EXTRA_BODY,
    )
    usage.consume(structured_usage)

    print(f"\nParsed {scenario.response_model.__name__}:")
    print(parsed.model_dump_json(indent=2))
    return Run(scenario, conversation_id, transcript, usage)


# --- Inspecting what landed on disk ------------------------------------------


def _short(value: str | None, width: int = 8) -> str:
    return (value or "-")[:width]


def _millis(start: str, end: str) -> float:
    return (
        datetime.fromisoformat(end) - datetime.fromisoformat(start)
    ).total_seconds() * 1000


def _span_detail(span: dict[str, Any]) -> str:
    attributes = span["attributes"]
    parts: list[str] = []
    if name := attributes.get(Attr.REQUEST_MODEL):
        parts.append(f"model={name}")
    if tool := attributes.get(Attr.TOOL_NAME):
        parts.append(f"tool={tool}")
    if fmt := attributes.get(Attr.RESPONSE_FORMAT):
        parts.append(f"format={fmt}")
    if Attr.USAGE_INPUT_TOKENS in attributes:
        parts.append(
            f"tokens={attributes.get(Attr.USAGE_INPUT_TOKENS)}"
            f"/{attributes.get(Attr.USAGE_OUTPUT_TOKENS, 0)}"
        )
    raw_cost = attributes.get(Attr.LANGFUSE_COST_DETAILS)
    if isinstance(raw_cost, str) and (cost := json.loads(raw_cost).get("total")):
        parts.append(f"cost=${cost:.6f}")
    parts.append(f"{_millis(span['start_time'], span['end_time']):.0f}ms")
    if span["parent_id"] is None and attributes.get(Attr.PARENT_TRACE_ID):
        parts.append(
            f"-> parent {_short(attributes.get(Attr.PARENT_TRACE_ID))}/"
            f"{_short(attributes.get(Attr.PARENT_SPAN_ID))}"
        )
    return "  ".join(parts)


def _walk(
    children: dict[str | None, list[dict[str, Any]]], parent: str | None, depth: int
) -> None:
    for span in sorted(children[parent], key=lambda s: s["start_time"]):
        print(f"    {'  ' * depth}{span['name']}  {_span_detail(span)}")
        _walk(children, span["context"]["span_id"], depth + 1)


def _print_tree(spans: list[dict[str, Any]]) -> None:
    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for span in spans:
        by_trace[span["context"]["trace_id"]].append(span)

    for trace_id, group in sorted(
        by_trace.items(), key=lambda item: min(s["start_time"] for s in item[1])
    ):
        print(f"  trace {_short(trace_id, 12)}")
        known = {span["context"]["span_id"] for span in group}
        children: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
        for span in group:
            parent = span["parent_id"] if span["parent_id"] in known else None
            children[parent].append(span)

        _walk(children, None, 0)


def inspect(runs: list[Run], spans: list[dict[str, Any]]) -> list[str]:
    """Print and verify the trace of every conversation; return problems found."""
    problems: list[str] = []
    print(f"\nTraces under {TRACES_DIR}:")
    print(
        f"{'conversation':<18}{'depth':>6}{'turns':>7}{'chats':>7}"
        f"{'tools':>7}{'structured':>12}{'traces':>8}"
    )

    for run in runs:
        conversation_id = run.conversation_id
        selected = [
            span
            for span in spans
            if span["attributes"].get(Attr.CONVERSATION_ID) == conversation_id
        ]
        turns = [span for span in selected if span["name"] == "invoke_agent"]
        chats = [span for span in selected if span["name"] == "chat"]
        tools = [span for span in selected if span["name"] == "execute_tool"]
        structured = [
            span for span in selected if span["name"] == "complete_structured"
        ]
        trace_ids = {span["context"]["trace_id"] for span in selected}
        depth = len(run.transcript)

        print(
            f"{run.scenario.key:<18}{depth:>6}{len(turns):>7}{len(chats):>7}"
            f"{len(tools):>7}{len(structured):>12}{len(trace_ids):>8}"
        )

        if not DEPTH_MIN <= depth <= DEPTH_MAX:
            problems.append(
                f"{run.scenario.key}: transcript depth {depth} outside {DEPTH_MIN}-{DEPTH_MAX}"
            )
        if len(turns) != len(run.scenario.turns):
            problems.append(
                f"{run.scenario.key}: expected {len(run.scenario.turns)} turn roots, found {len(turns)}"
            )
        if not tools:
            problems.append(f"{run.scenario.key}: the model never called a tool")
        if len(structured) != 1:
            problems.append(
                f"{run.scenario.key}: expected one complete_structured span, found {len(structured)}"
            )
        if len(chats) < len(turns):
            problems.append(
                f"{run.scenario.key}: fewer chat spans ({len(chats)}) than turns ({len(turns)})"
            )
        missing_id = [
            span["name"]
            for span in selected
            if Attr.CONVERSATION_ID not in span["attributes"]
        ]
        if missing_id:
            problems.append(
                f"{run.scenario.key}: spans without a conversation id: {sorted(set(missing_id))}"
            )
        unlinked = [
            span["name"]
            for span in turns
            if span is not turns[0] and Attr.PARENT_TRACE_ID not in span["attributes"]
        ]
        if unlinked:
            problems.append(
                f"{run.scenario.key}: turn roots without a parent edge: {len(unlinked)}"
            )

        print(f"\n{run.scenario.key} ({run.scenario.title})")
        _print_tree(selected)

    return problems


def _recent_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.jsonl"), key=lambda path: path.stat().st_mtime)


async def main() -> int:
    api_key = openrouter_api_key()
    if not api_key:
        print(
            f"{OPENROUTER_API_KEY_ENV} is not set; it is the key core.server uses for {OPENROUTER_URL}.",
            file=sys.stderr,
        )
        return 2

    provider = configure(TRACES_DIR)
    client = Client(
        base_url=OPENROUTER_URL, api_key=api_key, extra_body=OPENROUTER_EXTRA_BODY
    )
    try:
        runs: list[Run] = []
        for scenario in SCENARIOS:
            print(f"\n=== {scenario.key}: {scenario.title} ===")
            run = await run_scenario(client, scenario)
            rounds = _tool_rounds(run.transcript)
            print(
                f"\nconversation {run.conversation_id}: {len(run.transcript)} messages, "
                f"{rounds} tool round(s), {run.usage.total_tokens} tokens, "
                f"cost {run.usage.cost if run.usage.cost is not None else 'n/a'}"
            )
            runs.append(run)

        provider.force_flush()
        spans = read_spans(TRACES_DIR)
    finally:
        await client.aclose()
        shutdown()

    problems = inspect(runs, spans)
    files = _recent_files(TRACES_DIR)
    if files:
        print(f"\nNewest span file: {files[-1]}")
        print(f"Content blobs:    {TRACES_DIR / 'blobs_traces.db'}")

    if problems:
        print("\nTrace checks failed:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nAll trace checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
