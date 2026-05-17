"""Example raw-text transformation functions for Orbit"""

import re
import logging
from typing import Dict, Any
from pydantic import BaseModel, field_validator

from orbit.transformations.decorators import orbit_transformation_tool_mcp
from orbit.transformations.base import DataType

logger = logging.getLogger(__name__)


class SearchReplaceConfig(BaseModel):
    """Configuration for text search-and-replace"""

    pattern: str
    replacement: str
    use_regex: bool = False
    case_sensitive: bool = True
    # -1 means unlimited; positive int caps the number of substitutions
    max_replacements: int = -1

    @field_validator("max_replacements")
    @classmethod
    def validate_max_replacements(cls, v: int) -> int:
        if v < -1:
            raise ValueError("max_replacements must be -1 (unlimited) or a positive integer")
        return v


@orbit_transformation_tool_mcp(
    data_type=DataType.TEXT,
    description="Search and replace text in a raw text file, with optional regex support",
    transform_config=SearchReplaceConfig,
)
async def search_replace_text(
    resource_uri: str,
    pattern: str,
    replacement: str,
    use_regex: bool = False,
    case_sensitive: bool = True,
    max_replacements: int = -1,
    in_place: bool = False,
) -> Dict[str, Any]:
    """
    Search and replace text in a raw text file.

    Args:
        resource_uri: Path to the text file
        pattern: Literal string or regex pattern to search for
        replacement: Replacement string (supports regex backreferences when use_regex=True)
        use_regex: Whether pattern is a regular expression
        case_sensitive: Whether matching is case-sensitive (ignored for regex — embed (?i) instead)
        max_replacements: Maximum number of replacements (-1 for unlimited)
        in_place: Whether to overwrite the original file or write to a new path

    Returns:
        Summary with replacement count and character delta
    """
    try:
        with open(resource_uri, "r", encoding="utf-8") as f:
            original_text: str = f.read()

        count_limit: int = max_replacements if max_replacements != -1 else 0

        if use_regex:
            flags = 0
            compiled = re.compile(pattern, flags)
            if count_limit > 0:
                new_text, n_subs = compiled.subn(replacement, original_text, count=count_limit)
            else:
                new_text, n_subs = compiled.subn(replacement, original_text)
        else:
            if not case_sensitive:
                # Build a regex from the literal pattern so we can honour case-insensitivity
                escaped = re.escape(pattern)
                compiled = re.compile(escaped, re.IGNORECASE)
                if count_limit > 0:
                    new_text, n_subs = compiled.subn(replacement, original_text, count=count_limit)
                else:
                    new_text, n_subs = compiled.subn(replacement, original_text)
            else:
                # Pure string replacement — fastest path
                if count_limit > 0:
                    new_text = original_text.replace(pattern, replacement, count_limit)
                else:
                    new_text = original_text.replace(pattern, replacement)
                n_subs = (
                    (len(original_text) - len(original_text.replace(pattern, replacement)))
                    // max(len(pattern), 1)
                    if count_limit == 0
                    else count_limit
                )

        with open(resource_uri, "w", encoding="utf-8") as f:
            f.write(new_text)

        char_delta: int = len(new_text) - len(original_text)
        delta_sign: str = "+" if char_delta >= 0 else ""

        logger.info(
            "search_replace_text: %d substitution(s), char delta %s%d (use_regex=%s)",
            n_subs,
            delta_sign,
            char_delta,
            use_regex,
        )

        return {
            "type": "text",
            "text": (
                f"Replaced {n_subs} occurrence(s) of {repr(pattern)!s} with {repr(replacement)!s}. "
                f"Character count: {len(original_text)} -> {len(new_text)} ({delta_sign}{char_delta})."
            ),
        }

    except re.error as exc:
        logger.error("Invalid regex pattern '%s': %s", pattern, exc)
        return {"type": "text", "text": f"Regex error: {exc}"}
    except FileNotFoundError:
        logger.error("File not found: %s", resource_uri)
        return {"type": "text", "text": f"Error: file not found at {resource_uri!r}"}
    except Exception as exc:
        logger.error("search_replace_text failed: %s", exc)
        return {"type": "text", "text": f"Error: {exc}"}
