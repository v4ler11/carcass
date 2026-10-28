"""Response model for stage 1 overlay resolution."""

from enum import StrEnum

from pydantic import BaseModel, Field

from agent.citations import QuickCitation, VerifiedQuickCitation

COMMENTS_DESCRIPTION = (
    "Optional evidence notes for this determination, with an inline "
    "(c-XX)[...] citation tag for every factual claim and a matching "
    "top-level citations entry. Use this where the enum or boolean "
    "fields cannot carry a citable explanation, such as a negative "
    "check or an ambiguous boundary call."
)


class Confidence(StrEnum):
    high = "high"
    medium = "medium"
    low = "low"


class ERCategory(StrEnum):
    voirie_creation_amelioration = "voirie_creation_amelioration"
    equipements_espaces_publics = "equipements_espaces_publics"
    elargissements_voirie = "elargissements_voirie"


class GeometryType(StrEnum):
    line = "line"
    polygon = "polygon"
    point = "point"
    none = "none"  # no overlap, nothing to type


class ProximityToBoundary(StrEnum):
    central = "central"
    near_edge = "near_edge"
    ambiguous = "ambiguous"


class PresenceStatus(StrEnum):
    present = "present"
    absent = "absent"
    undetermined = "undetermined"


class ZoneDetermination(BaseModel):
    metadata_hypothesis: str = Field(
        ...,
        description=(
            "Zone code as supplied by the parcel metadata (GPU vector layer). "
            "This is a geometric/attribute-source hypothesis only, NOT a "
            "regulatory conclusion. Must be independently confirmed against "
            "the zoning map and legend before being treated as fact."
        ),
    )
    confirmed_zone: str | None = Field(
        default=None,
        description=(
            "Zone code as visually confirmed from the graphic regulation "
            "document (map + legend). Null if confirmation could not be "
            "made with reasonable confidence."
        ),
    )
    hypothesis_confirmed: bool = Field(
        ...,
        description=(
            "True if confirmed_zone matches metadata_hypothesis exactly. "
            "False if contradicted. Downstream nodes must branch on this "
            "field rather than string-comparing the two codes themselves."
        ),
    )
    fully_within_zone: bool = Field(
        ...,
        description=(
            "True if the entire parcel polygon falls within a single "
            "confirmed zone. False if the parcel appears to straddle a "
            "zone boundary, in which case this must also be logged in "
            "unresolved_flags at the top level."
        ),
    )
    confidence: Confidence = Field(
        ...,
        description=(
            "Self-assessed confidence in the zone confirmation, accounting "
            "for image resolution, legend symbol ambiguity, and "
            "georeference error near the parcel edge."
        ),
    )
    comments: str | None = Field(
        default=None,
        description=COMMENTS_DESCRIPTION,
    )


class EmplacementReserve(BaseModel):
    category: ERCategory = Field(
        ...,
        description=(
            "Which of the three supplied emplacement réservé category "
            "tables this entry corresponds to. One entry is required per "
            "category even when overlaps_parcel is False — absence must "
            "be explicit, not omitted."
        ),
    )
    overlaps_parcel: bool = Field(
        ...,
        description=(
            "True if any ER feature of this category intersects or "
            "touches the parcel boundary, based on visual inspection of "
            "the zoom imagery against the parcel outline."
        ),
    )
    geometry_type: GeometryType = Field(
        default=GeometryType.none,
        description=(
            "Shape of the overlapping ER feature, if any. A line-type "
            "reservation (e.g. a road-widening strip along frontage) "
            "affects the buildable envelope very differently from a "
            "polygon-type reservation covering interior parcel area. "
            "Required for downstream envelope synthesis; set to 'none' "
            "if overlaps_parcel is False."
        ),
    )
    reference_id: str | None = Field(
        default=None,
        description=(
            "The ER reference identifier from the corresponding table, "
            "if an overlap was found and a matching row was identified."
        ),
    )
    nature: str | None = Field(
        default=None,
        description=(
            "Plain-text description of what the ER reserves the land "
            "for (e.g. road widening, public equipment), taken from the "
            "matched table row."
        ),
    )
    affected_area_m2_estimate: float | None = Field(
        default=None,
        description=(
            "Rough estimate of parcel area affected by the overlap, in "
            "square meters, if visually estimable from the zoom imagery. "
            "Leave null and flag in unresolved_flags if precise "
            "measurement would require full GIS tooling beyond this "
            "assessment's scope."
        ),
    )
    affected_boundary_description: str | None = Field(
        default=None,
        description=(
            "Qualitative description of where on the parcel the overlap "
            "occurs (e.g. 'along northern frontage, ~3m depth'), used "
            "when a precise area estimate is not available."
        ),
    )
    confidence: Confidence = Field(
        ...,
        description="Self-assessed confidence in this overlap determination.",
    )
    comments: str | None = Field(
        default=None,
        description=COMMENTS_DESCRIPTION,
    )


class OAPSector(BaseModel):
    sector_name: str = Field(
        ...,
        description=(
            "Name of the specific OAP sector being checked (e.g. "
            "'Quartier Joliot Curie'), not just a generic OAP flag — a "
            "parcel could sit inside the wrong OAP sector if multiple "
            "exist in the document."
        ),
    )
    metadata_flag_count: int = Field(
        ...,
        description=(
            "Number of OAP-related prescription entries found in the "
            "source metadata for this parcel, carried over for reference. "
            "This is a geometric/attribute signal only, not confirmation."
        ),
    )
    confirmed_within_sector: bool = Field(
        ...,
        description=(
            "True if visual inspection of the OAP sector boundary on the "
            "zoning map confirms the parcel lies within it, independent "
            "of the metadata flag."
        ),
    )
    proximity_to_boundary: ProximityToBoundary = Field(
        ...,
        description=(
            "How close the parcel sits to the OAP sector's edge. "
            "'near_edge' or 'ambiguous' should lower downstream confidence "
            "in any OAP-derived constraint applied to this parcel."
        ),
    )
    confidence: Confidence = Field(
        ...,
        description="Self-assessed confidence in the sector confirmation.",
    )
    comments: str | None = Field(
        default=None,
        description=COMMENTS_DESCRIPTION,
    )


class PrescriptionCheck(BaseModel):
    type: str = Field(
        ...,
        description=(
            "Name of the graphic prescription category being checked, "
            "drawn from the general legend (e.g. 'espace_boise_classe', "
            "'alignment_line', 'protected_element'). Every category "
            "present in the legend should have a corresponding entry "
            "here, even if absent, to make the absence an explicit, "
            "auditable claim rather than a silent omission."
        ),
    )
    presence: PresenceStatus = Field(
        ...,
        description=(
            "Whether this prescription type is present on or touching the "
            "parcel, absent, or undetermined when the evidence is insufficient."
        ),
    )
    confidence: Confidence = Field(
        ...,
        description="Self-assessed confidence in this presence/absence determination.",
    )
    comments: str | None = Field(
        default=None,
        description=COMMENTS_DESCRIPTION,
    )


class GeoreferenceQualityFlag(BaseModel):
    residual_mean_m: float = Field(
        ...,
        description=(
            "Mean georeferencing residual error in meters, carried from "
            "the source metadata. Used to assess reliability of any "
            "boundary-proximity call made elsewhere in this object."
        ),
    )
    affects_boundary_calls: bool = Field(
        ...,
        description=(
            "True if the residual error is large enough relative to a "
            "close overlap/boundary call elsewhere in this object to cast "
            "doubt on that call (e.g. an ER overlap determined to be "
            "~1-2m from the parcel edge, comparable to the residual "
            "error itself)."
        ),
    )
    note: str | None = Field(
        default=None,
        description=(
            "Free-text explanation of which specific call(s) in this "
            "object are affected and why, if affects_boundary_calls is "
            "True."
        ),
    )


class ParcelContext(BaseModel):
    """
    Phase 1 output: confirmed zone and overlay context for a single parcel,
    derived from the graphic regulation document, legend, and emplacement
    réservé tables. This object gates all downstream rule-retrieval and
    program-admissibility reasoning — no later stage may assume a zone or
    overlay that is not explicitly confirmed here.
    """

    idu: str = Field(
        ...,
        description=(
            "Cadastral identifier of the parcel being assessed, carried "
            "through for traceability across all downstream pipeline "
            "stages and the final report."
        ),
    )
    zone: ZoneDetermination
    emplacements_reserves: list[EmplacementReserve] = Field(
        ...,
        description=(
            "One entry per supplied ER category table (voirie creation/"
            "amelioration, equipements/espaces publics, elargissements de "
            "voirie), regardless of whether an overlap was found."
        ),
    )
    oap_sector: OAPSector
    other_prescriptions_checklist: list[PrescriptionCheck] = Field(
        ...,
        description=(
            "Exhaustive, pre-enumerated list of all graphic prescription "
            "categories found in the general legend, each explicitly "
            "checked against the parcel. Must not be a freeform list of "
            "only what the agent happened to notice."
        ),
    )
    georeference_quality_flag: GeoreferenceQualityFlag
    unresolved_flags: list[str] = Field(
        default_factory=list,
        description=(
            "Any ambiguity, contradiction, or low-confidence determination "
            "elsewhere in this object that requires human review before "
            "this context is trusted downstream. Empty list only if every "
            "field above was resolved with medium-or-higher confidence."
        ),
    )
    citations: list[QuickCitation] = Field(
        ...,
        description=(
            "One entry per inline citation tag in this object. Each c_id "
            "must match exactly one inline tag, and each inline tag must "
            "have exactly one entry here."
        ),
    )


class VerifiedParcelContext(ParcelContext):
    """Stage 1 output whose citations carry independent verification verdicts."""

    citations: list[VerifiedQuickCitation] = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        ...,
        description=(
            "One entry per inline citation tag in this object, each carrying "
            "the original claim, verification instructions, and an independent "
            "confirmed/reasoning verdict."
        ),
    )
