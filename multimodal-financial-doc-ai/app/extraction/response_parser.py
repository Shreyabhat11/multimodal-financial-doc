"""
Parse a VLM's raw text response into a structured dict.

Why this needs to be its own module rather than a bare `json.loads(response)`: VLMs
reliably deviate from "output only JSON" in a small, predictable set of ways —
wrapping the JSON in ```json ... ``` fences, adding a leading sentence like "Here is
the extracted data:", or leaving a trailing comma before a closing bracket. Rather
than let each of these become a VisionModelResponseParsingError on the happy path,
we handle the known-common cases here and only raise once we've exhausted them —
that raise is a real signal (Phase 6 routes it to a per-page extraction failure, not
a silent empty result).
"""

from __future__ import annotations

import json
import re

from app.core.exceptions import VisionModelResponseParsingError

_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")


def _strip_code_fences(text: str) -> str:
    match = _CODE_FENCE_RE.search(text)
    return match.group(1) if match else text


def _extract_outermost_json_object(text: str) -> str:
    """Find the first '{' and its matching closing '}' by brace counting, so leading/
    trailing prose ("Here is the JSON:" / "Let me know if you need anything else.")
    around a valid JSON object doesn't break parsing."""
    start = text.find("{")
    if start == -1:
        raise VisionModelResponseParsingError(
            "No JSON object found in model response.", details={"raw_response_preview": text[:500]}
        )

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise VisionModelResponseParsingError(
        "Unbalanced braces — no matching closing '}' found for JSON object.",
        details={"raw_response_preview": text[:500]},
    )


def _fix_trailing_commas(json_text: str) -> str:
    return _TRAILING_COMMA_RE.sub(r"\1", json_text)


def parse_vlm_json_response(raw_text: str) -> dict:
    """Best-effort recovery of a JSON dict from raw VLM output text.

    Order of recovery attempts, cheapest/most-common first:
      1. Direct json.loads on the stripped text.
      2. Strip markdown code fences, then json.loads.
      3. Extract the outermost {...} object (handles leading/trailing prose), then
         json.loads, with a trailing-comma fixup applied as a last resort.

    Raises VisionModelResponseParsingError if none of these succeed.
    """
    if raw_text is None or not raw_text.strip():
        raise VisionModelResponseParsingError("Empty response from vision model.")

    candidate = raw_text.strip()

    for transform in (
        lambda t: t,
        _strip_code_fences,
        lambda t: _extract_outermost_json_object(_strip_code_fences(t)),
    ):
        try:
            text = transform(candidate)
            return json.loads(text)
        except (json.JSONDecodeError, VisionModelResponseParsingError):
            continue

    # Last resort: extract the object AND fix trailing commas.
    try:
        text = _fix_trailing_commas(_extract_outermost_json_object(_strip_code_fences(candidate)))
        return json.loads(text)
    except (json.JSONDecodeError, VisionModelResponseParsingError) as exc:
        raise VisionModelResponseParsingError(
            "Could not parse a valid JSON object from the vision model's response.",
            details={"raw_response_preview": raw_text[:1000]},
        ) from exc
