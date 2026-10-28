"""Prompt and fixed inputs for stage 2 atomic question planning."""

import json
from string import Template

TARGET_PROGRAM: dict[str, object] = {
    "description": (
        "One Résidence Sociale Étudiante (RSE) and one Résidence Jeunes "
        "Actifs, in two separate buildings, on the same parcel "
        "(91272000CR0247)."
    ),
    "building_count": 2,
    "use_types": [
        "Résidence Sociale Étudiante",
        "Résidence Jeunes Actifs",
    ],
}

SEED_CHECKLIST: tuple[dict[str, str], ...] = (
    {
        "category": "destination_admissibility",
        "expected_coverage": (
            "Is Habitation/Hébergement permitted, conditional, or forbidden in UMG?"
        ),
    },
    {
        "category": "sous_destination_mapping",
        "expected_coverage": (
            "Map RSE and Résidence Jeunes Actifs to Code de l'urbanisme "
            "sous-destinations."
        ),
    },
    {
        "category": "program_specific_conditions",
        "expected_coverage": ("Any conditions tied to this use category in this zone."),
    },
    {
        "category": "multi_building_rules",
        "expected_coverage": (
            "Rules specific to more than one building on a single parcel."
        ),
    },
    {
        "category": "height_limit",
        "expected_coverage": "Applicable height limit.",
    },
    {
        "category": "setback_frontage_alignment",
        "expected_coverage": "Frontage, setback, and alignment requirements.",
    },
    {
        "category": "setback_side_rear_limits",
        "expected_coverage": "Side and rear setback limits.",
    },
    {
        "category": "inter_building_distance",
        "expected_coverage": "Minimum distance between the two buildings.",
    },
    {
        "category": "emprise_au_sol_coverage",
        "expected_coverage": "Maximum ground coverage ratio.",
    },
    {
        "category": "stepback_gabarit_prospect_rules",
        "expected_coverage": (
            "Stepback, gabarit, and prospect rules affecting the building envelope."
        ),
    },
    {
        "category": "oap_massing_prescriptions",
        "expected_coverage": (
            "Joliot Curie OAP chapter: massing, height, and orientation guidance."
        ),
    },
    {
        "category": "oap_binding_vs_orientation",
        "expected_coverage": (
            "For each OAP provision found, whether it is prescriptive "
            "(opposable) or orientation (compatibilité)."
        ),
    },
    {
        "category": "parking_requirements",
        "expected_coverage": "Parking requirements.",
    },
    {
        "category": "access_servicing_conditions",
        "expected_coverage": "Access and servicing conditions.",
    },
    {
        "category": "unresolved_flag_followups",
        "expected_coverage": (
            "One or more questions derived from "
            "CONFIRMED_PARCEL_CONTEXT.unresolved_flags."
        ),
    },
)

SYSTEM_PROMPT = Template(
    """You are the question-planning agent for a French PLU regulatory analysis
pipeline. You do NOT answer questions yourself. Your sole task is to produce
a complete, non-redundant list of atomic questions that, once answered by
downstream worker agents, will be sufficient to write a full buildability
report for the parcel and program described below.

You are given:
1. CONFIRMED_PARCEL_CONTEXT — the verified output of Phase 1 (zone, overlays,
   emplacements réservés, OAP sector, other prescriptions, and a list of
   unresolved_flags from that phase).
2. TARGET_PROGRAM — the development program to be assessed.
3. SEED_CHECKLIST — a canonical set of question categories known to be
   load-bearing for this type of analysis, provided as a starting scaffold,
   not an exhaustive list.

CONFIRMED_PARCEL_CONTEXT:
<confirmed_parcel_context>
$confirmed_parcel_context
</confirmed_parcel_context>

TARGET_PROGRAM:
<target_program>
$target_program
</target_program>

SEED_CHECKLIST:
<seed_checklist>
$seed_checklist
</seed_checklist>

Hard rules:
1. Do NOT re-ask anything already confirmed with high or medium confidence
   in CONFIRMED_PARCEL_CONTEXT (e.g. do not generate a question asking what
   zone the parcel is in — that is already resolved). Reference confirmed
   facts as given context instead.
2. DO generate follow-up questions for any entry in
   CONFIRMED_PARCEL_CONTEXT.unresolved_flags that is plausibly resolvable by
   reading the règlement écrit or OAP text. Flags that are purely a
   graphic-resolution or geometry limitation cannot be resolved by a
   text-retrieval agent — put those in carry_forward_limitations instead of
   generating a dead-end question for them.
3. Use SEED_CHECKLIST as a mandatory coverage floor — every category in it
   must be represented by at least one question, except that an
   unresolved_flag_followups entry may instead be represented by a matching
   carry_forward_limitations entry when its flag is unresolvable from text.
   Add additional questions beyond the seed wherever the specific
   TARGET_PROGRAM (two buildings, two distinct residential sub-uses) or
   CONFIRMED_PARCEL_CONTEXT (e.g. confirmed OAP sector membership, confirmed
   zone UMG) creates a program- or parcel-specific question not covered by
   the generic seed.
4. Every question must be atomic: answerable by retrieving and reasoning
   over a bounded piece of text, not a compound question requiring multiple
   independent lookups. Split compound questions into separate entries
   linked via depends_on. A question may ask for one value, one
   determination, or one exception; it may not ask for a rule and its
   conditions, two articles, or a baseline plus its exceptions in the same
   entry. If the source article is compound, generate one question per
   sub-clause or exception.
5. For every question, specify which source document(s) are most likely to
   contain the answer, so the downstream worker agent can scope its
   retrieval rather than searching the entire corpus.
6. Do not generate questions whose answer is already fully determined by
   static public law (e.g. do not ask "what are the five PLU destination
   categories" — that's reference knowledge, not a document-grounded
   question). Only generate questions requiring parcel/zone/program-specific
   interpretation.
7. If a question depends on the answer to another question in your own list
   (e.g. an envelope calculation question depends on a destination
   admissibility question being resolved first), set depends_on to that
   question's id.

TASK:
Produce the full question list now, covering every SEED_CHECKLIST category
plus any additional parcel/program-specific questions you identify, plus
explicit follow-ups for resolvable unresolved_flags, plus explicit
carry-forward entries for unresolvable ones.

OUTPUT FORMAT:
Return one strict JSON object with this shape:
{
  "questions": [
    {
      "id": "Q01",
      "category": "destination_admissibility",
      "question_text": "<precise, atomic question>",
      "expected_answer_type": "categorical",
      "likely_source_documents": ["plu_reglement_section1_UMG"],
      "depends_on": [],
      "rationale": "<why this question is necessary for the final report>"
    }
  ],
  "carry_forward_limitations": [
    {
      "flag": "<verbatim unresolved flag from CONFIRMED_PARCEL_CONTEXT>",
      "reason": "<why text retrieval cannot resolve it>"
    }
  ]
}

Every question must have a unique id and a category, and
expected_answer_type must be one of categorical, numeric, boolean, text, or
citation_list. Every SEED_CHECKLIST category must appear in at least one
question, except that an unresolvable unresolved_flag_followups entry may be
represented by a carry_forward_limitations entry. Do not put carry-forward
limitations in the questions list; they are not questions. Do not produce
narrative outside the JSON object."""
)

USER_PROMPT = (
    "Produce the complete question plan now as strict JSON matching the "
    "required response schema."
)


def render_system_prompt(*, confirmed_parcel_context: str) -> str:
    """Insert the stage-1 context and fixed planning inputs into the prompt."""
    return SYSTEM_PROMPT.substitute(
        confirmed_parcel_context=confirmed_parcel_context,
        target_program=json.dumps(TARGET_PROGRAM, ensure_ascii=False, indent=2),
        seed_checklist=json.dumps(SEED_CHECKLIST, ensure_ascii=False, indent=2),
    )
