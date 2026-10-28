"""PDF -> Markdown conversion via Mistral Document AI OCR."""

from core.config import DEFAULT_OCR_MODEL

from .document import ImageRef, OcrDocument, Page, TableRef
from .markdown import write_document
from .mistral_ocr import (
    OcrError,
    OcrSettings,
    RequestPlan,
    ocr_pdf,
    plan_requests,
)

__all__ = [
    "DEFAULT_OCR_MODEL",
    "ImageRef",
    "OcrDocument",
    "OcrError",
    "OcrSettings",
    "Page",
    "RequestPlan",
    "TableRef",
    "ocr_pdf",
    "plan_requests",
    "write_document",
]
