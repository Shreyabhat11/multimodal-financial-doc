"""Data model for a single processed page.

Kept as a plain dataclass (not a Pydantic model) because it carries a live PIL.Image
object — Pydantic would need a custom arbitrary-type config for that, and this object
never crosses a serialization boundary (it lives entirely in-process between
preprocessing and the VLM call). Anything that DOES need to cross a boundary
(LangGraph state, API responses, DB rows) uses page_number/metadata only, not the
image bytes themselves — see PageMetadata below.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image


@dataclass
class PageImage:
    """An in-memory rendered page, ready to be sent to a vision-language model."""

    page_number: int  # 1-indexed, matches how humans refer to "page 3"
    image: Image.Image
    source_dpi: int
    original_width: int
    original_height: int
    rotation_applied_degrees: int = 0  # 0, 90, 180, or 270 — auto-correction applied, if any
    was_rotation_detected: bool = False

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    def to_metadata(self) -> "PageMetadata":
        return PageMetadata(
            page_number=self.page_number,
            width=self.width,
            height=self.height,
            source_dpi=self.source_dpi,
            rotation_applied_degrees=self.rotation_applied_degrees,
        )


@dataclass
class PageMetadata:
    """Serializable subset of PageImage — safe to store in DB rows / LangGraph state /
    API responses (no raw image bytes)."""

    page_number: int
    width: int
    height: int
    source_dpi: int
    rotation_applied_degrees: int = 0


@dataclass
class PreprocessedDocument:
    """The full output of the document_processing stage for one uploaded file."""

    document_id: str
    original_filename: str
    page_count: int
    file_size_bytes: int
    pages: list[PageImage] = field(default_factory=list)

    @property
    def page_metadata_list(self) -> list[PageMetadata]:
        return [p.to_metadata() for p in self.pages]
