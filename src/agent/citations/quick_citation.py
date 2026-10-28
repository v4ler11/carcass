"""Quick-citation model and guardrail."""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

_TAG = re.compile(r"\((c-\d{2,})\)\[|\[(c-\d{2,})\]\(")
_PREFIXES = ("TEXT |", "VISUAL |", "TABLE |", "METADATA |")


class QuickCitation(BaseModel):
    """One citation, addressed by the `c_id` used in the inline tag."""

    model_config = ConfigDict(populate_by_name=True)

    claim: str = Field(
        description=(
            "The factual claim supported by this citation, stated without the "
            "inline tag. One distinct claim per c_id. Use this for cited "
            "numeric values, zone/overlay identifications, admissibility "
            "judgments, rules that apply or do not apply, metadata-sourced "
            "assertions, and negative claims about what was checked. Do not "
            "put meta-commentary, process notes, or confidence/ambiguity "
            "statements here."
        )
    )
    id: str = Field(alias="c_id", pattern=r"^c-\d{2,}$")
    verification_instructions: str = Field(
        description=(
            "How to verify the claim. Must start with one of these prefixes: "
            "TEXT | <doc_filename> | page=<N> | quote='<verbatim quote>'; "
            "VISUAL | <image_filename> | bbox=[x0,y0,x1,y1] | <what to compare "
            "against>; TABLE | <table_filename> | rows=[<row_ids or "
            "'none_found'>] | <what was checked>; METADATA | "
            "<field_path_in_metadata_json> | <value cited>. Name only inputs "
            "provided in this call; for a negative claim, state the scope "
            "checked."
        )
    )

    @field_validator("verification_instructions")
    @classmethod
    def _check_prefix(cls, value: str) -> str:
        if not value.startswith(_PREFIXES):
            raise ValueError(
                "verification_instructions must start with one of: "
                + ", ".join(_PREFIXES)
            )
        return value


def has_citation_tag(text: str) -> bool:
    """Whether `text` contains at least one inline citation tag."""
    return _TAG.search(text) is not None


def citation_problems(
    text: str,
    citations: Sequence[QuickCitation],
) -> list[str]:
    """Everything that makes the inline tags and `citations` inconsistent."""
    text_ids = [first or second for first, second in _TAG.findall(text)]
    citation_ids = [citation.id for citation in citations]
    problems: list[str] = []

    duplicated_tags = sorted({cid for cid in text_ids if text_ids.count(cid) > 1})
    if duplicated_tags:
        problems.append(
            "citation ids used more than once in the text: "
            + ", ".join(duplicated_tags)
        )

    duplicated_citations = sorted(
        {cid for cid in citation_ids if citation_ids.count(cid) > 1}
    )
    if duplicated_citations:
        problems.append(
            "citation ids present more than once in citations[]: "
            + ", ".join(duplicated_citations)
        )

    missing = sorted(set(text_ids) - set(citation_ids))
    if missing:
        problems.append(
            "inline tags without a citations[] entry: " + ", ".join(missing)
        )

    unused = sorted(set(citation_ids) - set(text_ids))
    if unused:
        problems.append(
            "citations[] entries without an inline tag: " + ", ".join(unused)
        )

    return problems


def cited_text(value: Any) -> str:
    """All textual claims in a structured output, excluding `citations`."""
    if isinstance(value, BaseModel):
        return cited_text(value.model_dump(exclude={"citations"}))
    if isinstance(value, Mapping):
        return "\n".join(
            cited_text(item) for key, item in value.items() if key != "citations"
        )
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return "\n".join(cited_text(item) for item in value)
    return ""


def validate_quick_citations(answer: BaseModel) -> list[str]:
    """Guardrail: inline tags and `citations` must agree exactly."""
    citations = getattr(answer, "citations", None)
    if not isinstance(citations, list):
        return [f"{type(answer).__name__} must expose a citations list"]
    return citation_problems(cited_text(answer), citations)
