CITATION_PROMPT = """CITATION RULES (apply to every claim you make):

FORMAT:
Every factual claim ends with an inline tag: (c-XX)[quoted text or
visual description], where XX is a zero-padded, sequential ID, unique
within this response (c-01, c-02, c-03...).

In your structured output, add exactly one entry per tag to the
`citations` array with three fields, in this order: `claim`, `c_id`, and
`verification_instructions`.

`claim` is the factual claim supported by the tag, stated without the
inline tag. One distinct claim per c_id. Do not put meta-commentary,
process notes, or confidence/ambiguity statements in `claim`.

verification_instructions MUST start with one of these prefixes:
  - "TEXT | <doc_filename> | page=<N> | quote='<verbatim quote>'"
  - "VISUAL | <image_filename> | bbox=[x0,y0,x1,y1] | <what to compare against>"
  - "TABLE | <table_filename> | rows=[<row_ids or 'none_found'>] | <what was checked>"
  - "METADATA | <field_path_in_metadata_json> | <value cited>"

WHAT REQUIRES A CITATION:
- Any numeric value (height, setback, coverage ratio, distance, area)
- Any zone/overlay/sector identification or confirmation
- Any admissibility judgment (permitted/restricted/forbidden)
- Any statement that a rule, prescription, or constraint applies —
  OR explicitly does not apply
- Any claim sourced from parcel_metadata.json (tag as METADATA — this
  marks it as geometric-source, not regulatory, per your source_type rules)

NEGATIVE CLAIMS STILL REQUIRE CITATION:
"No emplacement réservé intersects this parcel" is a claim and must cite
WHAT was checked (e.g. TABLE | er_table_x.md | rows=[ER-01,ER-02,ER-03]
| confirmed no geometry overlap) — never state an absence without citing
the scope of what was examined. An absence claim with no citation is
treated as a failure (unverifiable negative).

WHAT DOES NOT REQUIRE A CITATION:
- Meta-commentary, restating the question, describing your own process
- Confidence/ambiguity statements about your own reasoning (these go in
  a separate `unresolved_flags` field, not a citation)
- Pure arithmetic performed on already-cited values (cite the inputs
  once; the derived calculation doesn't need its own citation, show
  the formula instead)

WHAT YOU MUST NEVER DO:
- Never cite a document, page, or table you were not given in this
  call's inputs
- Never reuse a c_id for two different claims
- Never leave an inline tag without a matching citations[] entry, or
  vice versa
- Never state "no evidence found" without specifying what you searched
  (same rule as negative claims above)
- If you cannot find support for a claim you believe is true, do not
  cite anything — instead state it under `unresolved_flags` with
  reasoning, uncited
"""
