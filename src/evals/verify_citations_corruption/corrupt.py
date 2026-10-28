"""Corrupt citations with one LLM call per answer and corruption mode."""

import json
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from agent.citations import QuickCitation
from chat.clients import Client
from chat.structured import complete_structured


class Mode(StrEnum):
    claim_factual = "corrupt_claim_factual"
    claim_complete = "corrupt_claim_complete"
    verification_instructions = "corrupt_verification_instructions"


_FORMAT = """You receive a JSON list of citations: {c_id, claim, verification_instructions}.
Return one item per input c_id, in the same order, with the same c_id and the
rewritten value in `value`. Keep the language of the original. Return only the
structured output."""

PROMPTS: dict[Mode, str] = {
    Mode.claim_factual: (
        "You build negative test cases for a citation verifier. Rewrite each "
        "`claim` so that it is factually WRONG while staying very close to the "
        "original: same topic, same sentence structure, same vocabulary. Change "
        "only concrete factual details: numbers, thresholds, units, article or "
        "zone names, the polarity of a rule (permitted/forbidden, minimum/"
        "maximum). The result must look as plausible as the original.\n\n"
        + _FORMAT
    ),
    Mode.claim_complete: (
        "You build negative test cases for a citation verifier. Replace each "
        "`claim` with a statement that is WRONG and DISTANT from the original: "
        "a different subject within French urban-planning regulation, with a "
        "different rule, different numbers and as little vocabulary in common "
        "with the original as possible. It must still read like a plausible "
        "claim from a PLU document.\n\n" + _FORMAT
    ),
    Mode.verification_instructions: (
        "You build negative test cases for a citation verifier. Rewrite each "
        "`verification_instructions` so that it points to the WRONG evidence "
        "while the claim stays unchanged. Keep the exact format and the leading "
        "tag (`TEXT |`, `VISUAL |`, `TABLE |` or `METADATA |`) and the same "
        "fields, but change what is pointed at: another document page, a "
        "different quote that does not support the claim, another table row, "
        "another image or another metadata field. The result must look as "
        "plausible as the original.\n\n" + _FORMAT
    ),
}


class CorruptedValue(BaseModel):
    c_id: str = Field(..., pattern=r"^c-\d{2,}$")
    value: str = Field(..., min_length=1)


class CorruptedValues(BaseModel):
    items: list[CorruptedValue]


async def corrupt_citations(
    *,
    client: Client,
    model: str,
    citations: Sequence[QuickCitation],
    mode: Mode,
    metadata: Mapping[str, Any] | None = None,
) -> list[QuickCitation]:
    """Citations with the field named by `mode` rewritten by the model."""
    payload = [
        {
            "c_id": c.id,
            "claim": c.claim,
            "verification_instructions": c.verification_instructions,
        }
        for c in citations
    ]
    parsed, *_ = await complete_structured(
        client=client,
        model=model,
        messages=[
            {"role": "system", "content": PROMPTS[mode]},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_model=CorruptedValues,
        metadata={**(metadata or {}), "step": "corrupt", "variant": mode.value},
    )

    values = {item.c_id: item.value for item in parsed.items}
    expected = [c.id for c in citations]
    if sorted(values) != sorted(expected) or len(parsed.items) != len(expected):
        raise ValueError(
            f"{mode.value}: corruptor returned c_ids {sorted(values)}, "
            f"expected {sorted(expected)}"
        )

    field = (
        "verification_instructions"
        if mode is Mode.verification_instructions
        else "claim"
    )
    return [
        QuickCitation.model_validate(
            {
                "c_id": c.id,
                "claim": c.claim,
                "verification_instructions": c.verification_instructions,
                field: values[c.id],
            }
        )
        for c in citations
    ]
