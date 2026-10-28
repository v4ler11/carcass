import json

import pytest
from openai.types import CompletionUsage

from chat.usage import ChatUsage


def test_reported_zero_counters_are_emitted() -> None:
    attributes = ChatUsage(model="m", prompt_tokens=10, cached_tokens=0).to_attributes()

    assert attributes["gen_ai.usage.cache_read.input_tokens"] == 0


def test_unreported_counters_are_omitted() -> None:
    attributes = ChatUsage(model="m", prompt_tokens=10, cached_tokens=None).to_attributes()

    assert "gen_ai.usage.cache_read.input_tokens" not in attributes


def test_attributes_round_trip_including_zero() -> None:
    usage = ChatUsage(
        model="m",
        prompt_tokens=10,
        completion_tokens=2,
        total_tokens=12,
        cached_tokens=0,
        reasoning_tokens=3,
        cost=0.5,
        prompt_cost=0.3,
        completion_cost=0.2,
    )

    assert ChatUsage.from_attributes(usage.to_attributes()) == usage


def test_cost_details_are_emitted() -> None:
    attributes = ChatUsage(
        model="m", cost=0.5, prompt_cost=0.3, completion_cost=0.2
    ).to_attributes()

    assert json.loads(attributes["langfuse.observation.cost_details"]) == {
        "total": 0.5,
        "input": 0.3,
        "output": 0.2,
    }


def test_openrouter_cost_details_are_read() -> None:
    usage = CompletionUsage.model_validate(
        {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "cost": 0.5,
            "cost_details": {
                "upstream_inference_prompt_cost": 0.3,
                "upstream_inference_completions_cost": 0.2,
            },
        }
    )

    chat_usage = ChatUsage.from_openai(usage, "m")

    assert (chat_usage.cost, chat_usage.prompt_cost, chat_usage.completion_cost) == (0.5, 0.3, 0.2)


def test_consume_accumulates_cost() -> None:
    total = ChatUsage(model="m")
    total.consume(ChatUsage(model="m", cost=0.5, prompt_cost=0.3, completion_cost=0.2))
    total.consume(ChatUsage(model="m", cost=0.1, prompt_cost=0.05, completion_cost=0.05))

    assert (total.cost, total.prompt_cost, total.completion_cost) == pytest.approx((0.6, 0.35, 0.25))
