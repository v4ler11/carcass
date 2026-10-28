"""Locate a cadastral parcel on the PLU graphic-regulation sheet, end to end.

Stages: ``geo`` (BAN / cadastre / GPU), ``overlay`` (sheet georeference and
placement), ``panels`` (annex panels), ``tables`` (OCR of the reserve tables).
``pipeline`` chains them; ``ingest_pipeline`` runs the sheet stages as part of
the full corpus build, and ``python -m ingest.plu_graph.pipeline`` runs them
standalone.
"""


def __getattr__(name: str):
    """Import the pipeline on demand, so ``python -m ingest.plu_graph.geo`` stays clean."""
    if name == "main":
        from .pipeline import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
