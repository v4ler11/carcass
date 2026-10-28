import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from openai.types import CompletionUsage
from openai.types.decision import Usage as DecisionUsage

from tracing.attributes import Attr

_COUNTERS = (
    "cached_tokens",
    "cache_write_tokens",
    "text_input_tokens",
    "image_input_tokens",
    "audio_input_tokens",
    "text_output_tokens",
    "audio_output_tokens",
    "reasoning_tokens",
    "accepted_prediction_tokens",
    "rejected_prediction_tokens",
)

_ATTRIBUTES = {
    "cached_tokens": Attr.USAGE_CACHE_READ_INPUT_TOKENS,
    "cache_write_tokens": Attr.USAGE_CACHE_WRITE_INPUT_TOKENS,
    "text_input_tokens": Attr.USAGE_TEXT_INPUT_TOKENS,
    "image_input_tokens": Attr.USAGE_IMAGE_INPUT_TOKENS,
    "audio_input_tokens": Attr.USAGE_AUDIO_INPUT_TOKENS,
    "text_output_tokens": Attr.USAGE_TEXT_OUTPUT_TOKENS,
    "audio_output_tokens": Attr.USAGE_AUDIO_OUTPUT_TOKENS,
    "reasoning_tokens": Attr.USAGE_REASONING_OUTPUT_TOKENS,
}


def _detail(details: Any, name: str) -> int | None:
    return getattr(details, name, None) if details is not None else None


def _cost_detail(details: Any, name: str) -> float | None:
    value = details.get(name) if isinstance(details, dict) else None
    return float(value) if isinstance(value, int | float) else None


@dataclass(slots=True)
class ChatUsage:
    model: str

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    text_input_tokens: int | None = None
    image_input_tokens: int | None = None
    audio_input_tokens: int | None = None
    text_output_tokens: int | None = None
    audio_output_tokens: int | None = None
    reasoning_tokens: int | None = None
    accepted_prediction_tokens: int | None = None
    rejected_prediction_tokens: int | None = None

    cost: float | None = None
    prompt_cost: float | None = None
    completion_cost: float | None = None

    @classmethod
    def from_openai(cls, usage: CompletionUsage | None, model: str) -> Self:
        if usage is None:
            return cls(model=model)

        prompt = usage.prompt_tokens_details
        completion = usage.completion_tokens_details
        cost_details = getattr(usage, "cost_details", None)

        return cls(
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cached_tokens=_detail(prompt, "cached_tokens"),
            cache_write_tokens=_detail(prompt, "cache_write_tokens"),
            text_input_tokens=_detail(prompt, "text_tokens"),
            image_input_tokens=_detail(prompt, "image_tokens"),
            audio_input_tokens=_detail(prompt, "audio_tokens"),
            text_output_tokens=_detail(completion, "text_tokens"),
            audio_output_tokens=_detail(completion, "audio_tokens"),
            reasoning_tokens=_detail(completion, "reasoning_tokens"),
            accepted_prediction_tokens=_detail(completion, "accepted_prediction_tokens"),
            rejected_prediction_tokens=_detail(completion, "rejected_prediction_tokens"),
            cost=getattr(usage, "cost", None),
            prompt_cost=_cost_detail(cost_details, "upstream_inference_prompt_cost"),
            completion_cost=_cost_detail(cost_details, "upstream_inference_completions_cost"),
        )

    @classmethod
    def from_decision(cls, usage: DecisionUsage, model: str) -> Self:
        """Usage of one Decisions request: input is charged, output tokens are free."""
        return cls(
            model=model,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cached_tokens=usage.input_tokens_details.cached_tokens,
            cache_write_tokens=usage.input_tokens_details.cache_write_tokens,
            reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
        )

    @classmethod
    def from_attributes(cls, attributes: Mapping[str, Any]) -> Self:
        """Rebuild usage from an inference span: the inverse of `to_attributes`."""
        prompt = attributes.get(Attr.USAGE_INPUT_TOKENS, 0)
        completion = attributes.get(Attr.USAGE_OUTPUT_TOKENS, 0)
        costs: dict[str, Any] = {}
        raw_costs = attributes.get(Attr.LANGFUSE_COST_DETAILS)
        if isinstance(raw_costs, str):
            try:
                parsed = json.loads(raw_costs)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                costs = parsed
        return cls(
            model=attributes.get(Attr.REQUEST_MODEL, ""),
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=attributes.get(Attr.USAGE_TOTAL_TOKENS, prompt + completion),
            cost=costs.get("total"),
            prompt_cost=costs.get("input"),
            completion_cost=costs.get("output"),
            **{field: attributes.get(key) for field, key in _ATTRIBUTES.items()},
        )

    def consume(self, usage: Self) -> None:
        if usage.model != self.model:
            raise ValueError(f"cannot consume usage from {usage.model!r} into {self.model!r}")

        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens

        for field in (*_COUNTERS, "cost", "prompt_cost", "completion_cost"):
            value = getattr(usage, field)
            if value is not None:
                setattr(self, field, (getattr(self, field) or 0) + value)

    def to_attributes(self) -> dict[str, Any]:
        """The usage contribution to a `gen_ai` inference span.

        Only what this object knows is emitted. `gen_ai.operation.name`,
        `gen_ai.provider.name`, `gen_ai.response.*` and the request sampling
        parameters belong to the span builder, which also holds the endpoint,
        the `ChatCompletion` and the call parameters.
        """
        attributes: dict[str, Any] = {
            Attr.REQUEST_MODEL: self.model,
            Attr.USAGE_INPUT_TOKENS: self.prompt_tokens,
            Attr.USAGE_OUTPUT_TOKENS: self.completion_tokens,
        }
        for field, key in _ATTRIBUTES.items():
            value = getattr(self, field)
            if value is not None:
                attributes[key] = value
        costs = {
            key: value
            for key, value in (
                ("total", self.cost),
                ("input", self.prompt_cost),
                ("output", self.completion_cost),
            )
            if value is not None
        }
        if costs:
            attributes[Attr.LANGFUSE_COST_DETAILS] = json.dumps(
                costs, ensure_ascii=False, sort_keys=True
            )
        return attributes
