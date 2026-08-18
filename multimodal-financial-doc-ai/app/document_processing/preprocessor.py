"""
Per-page image preprocessing.

This runs AFTER pdf_loader has rendered each page to a PIL Image and BEFORE the image
is sent to a vision-language model. Kept as a separate module from pdf_loader
deliberately: pdf_loader's job is "get pixels out of the PDF correctly" (rotation,
resolution); this module's job is "make those pixels a good VLM input" (bounding size,
normalizing mode, mild contrast correction for scanned documents). Different concerns,
different failure modes, easier to test in isolation.
"""

from __future__ import annotations

from PIL import Image, ImageEnhance, ImageOps

# Most VLM processors (Qwen2-VL, LLaVA) internally resize to a model-specific max
# resolution anyway, and very large page images (e.g. 300+ DPI renders of A3 pages)
# needlessly inflate memory and upload/preprocessing time without adding extraction
# quality. Capping the longest edge here keeps memory bounded and consistent
# regardless of what DPI the page was rendered at.
MAX_LONGEST_EDGE_PX = 2000

# A floor as well: an under-sized image (e.g. a low-DPI scan) makes small print and
# thin table gridlines unreadable to the VLM. We don't upscale by default since
# upscaling doesn't add real information and can introduce artifacts the model might
# hallucinate structure from — but we expose the constant so callers can opt in.
MIN_RECOMMENDED_LONGEST_EDGE_PX = 900


def normalize_image_mode(image: Image.Image) -> Image.Image:
    """Ensure the image is RGB. PDFs occasionally render as CMYK or with an alpha
    channel depending on embedded color profiles; downstream VLM processors expect RGB."""
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def resize_to_bounds(image: Image.Image, *, max_longest_edge: int = MAX_LONGEST_EDGE_PX) -> Image.Image:
    """Downscale an image so its longest edge is at most `max_longest_edge`, preserving
    aspect ratio. No-op if the image is already within bounds."""
    longest_edge = max(image.width, image.height)
    if longest_edge <= max_longest_edge:
        return image
    scale = max_longest_edge / longest_edge
    new_size = (round(image.width * scale), round(image.height * scale))
    return image.resize(new_size, Image.LANCZOS)


def enhance_scanned_contrast(image: Image.Image, *, factor: float = 1.15) -> Image.Image:
    """Mild contrast boost, primarily to help low-quality scanned statements where
    faint print/table gridlines would otherwise be hard to distinguish from background.
    Factor is intentionally conservative (1.15, not more) — over-enhancing risks
    clipping and can distort thin gridlines that carry table-structure information the
    VLM relies on to correctly segment transaction rows.
    """
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def auto_orient(image: Image.Image) -> Image.Image:
    """Apply any EXIF orientation tag, for the (uncommon but possible) case of a
    photographed statement page rather than a rendered PDF page. No-op for PDF-derived
    images, which carry no EXIF orientation."""
    return ImageOps.exif_transpose(image) or image


def preprocess_page_image(
    image: Image.Image,
    *,
    max_longest_edge: int = MAX_LONGEST_EDGE_PX,
    apply_contrast_enhancement: bool = True,
) -> Image.Image:
    """Full per-page preprocessing pipeline, applied in a fixed, documented order:

    1. auto_orient      — correct EXIF orientation, if present
    2. normalize_image_mode — force RGB
    3. resize_to_bounds  — bound memory/latency
    4. enhance_scanned_contrast — mild contrast boost (optional, on by default)
    """
    image = auto_orient(image)
    image = normalize_image_mode(image)
    image = resize_to_bounds(image, max_longest_edge=max_longest_edge)
    if apply_contrast_enhancement:
        image = enhance_scanned_contrast(image)
    return image
