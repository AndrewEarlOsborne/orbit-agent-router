"""Tests for text search-and-replace transformation"""

import os
import tempfile
import pytest

from orbit.transformations.base import DataType, get_registry
from orbit.transformations.examples.text_transforms import search_replace_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tmp(content: str) -> str:
    """Write content to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_registered_in_global_registry(self) -> None:
        registry = get_registry()
        meta = registry.get(DataType.TEXT, "search_replace_text")
        assert meta is not None
        assert meta.name == "search_replace_text"
        assert meta.data_type == DataType.TEXT

    def test_orbit_metadata_attached(self) -> None:
        assert getattr(search_replace_text, "_orbit_transform", False) is True
        assert search_replace_text._orbit_data_type == DataType.TEXT


# ---------------------------------------------------------------------------
# Literal search-and-replace
# ---------------------------------------------------------------------------

class TestLiteralReplace:
    @pytest.mark.asyncio
    async def test_basic_replacement(self) -> None:
        path = _write_tmp("hello world hello")
        try:
            result = await search_replace_text(
                resource_uri=path,
                pattern="hello",
                replacement="goodbye",
                in_place=True,
            )
            assert result["type"] == "text"
            assert "2" in result["text"]  # 2 occurrences replaced
            with open(path, encoding="utf-8") as f:
                assert f.read() == "goodbye world goodbye"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_no_match_returns_zero_replacements(self) -> None:
        path = _write_tmp("nothing to match here")
        try:
            result = await search_replace_text(
                resource_uri=path,
                pattern="ZZZMISSING",
                replacement="nope",
                in_place=True,
            )
            assert "0" in result["text"]
            with open(path, encoding="utf-8") as f:
                assert f.read() == "nothing to match here"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_max_replacements_cap(self) -> None:
        path = _write_tmp("a a a a a")
        try:
            await search_replace_text(
                resource_uri=path,
                pattern="a",
                replacement="b",
                max_replacements=2,
                in_place=True,
            )
            with open(path, encoding="utf-8") as f:
                content = f.read()
            assert content.count("b") == 2
            assert content.count("a") == 3
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_case_insensitive_literal(self) -> None:
        path = _write_tmp("Hello HELLO hello")
        try:
            await search_replace_text(
                resource_uri=path,
                pattern="hello",
                replacement="hi",
                case_sensitive=False,
                in_place=True,
            )
            with open(path, encoding="utf-8") as f:
                content = f.read()
            assert content == "hi hi hi"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_case_sensitive_literal(self) -> None:
        path = _write_tmp("Hello HELLO hello")
        try:
            await search_replace_text(
                resource_uri=path,
                pattern="hello",
                replacement="hi",
                case_sensitive=True,
                in_place=True,
            )
            with open(path, encoding="utf-8") as f:
                content = f.read()
            # Only the lowercase "hello" should be replaced
            assert "Hello" in content
            assert "HELLO" in content
            assert content.count("hi") == 1
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Regex search-and-replace
# ---------------------------------------------------------------------------

class TestRegexReplace:
    @pytest.mark.asyncio
    async def test_regex_digit_removal(self) -> None:
        path = _write_tmp("abc123def456")
        try:
            await search_replace_text(
                resource_uri=path,
                pattern=r"\d+",
                replacement="",
                use_regex=True,
                in_place=True,
            )
            with open(path, encoding="utf-8") as f:
                assert f.read() == "abcdef"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_regex_backreference(self) -> None:
        path = _write_tmp("2024-01-15")
        try:
            await search_replace_text(
                resource_uri=path,
                pattern=r"(\d{4})-(\d{2})-(\d{2})",
                replacement=r"\3/\2/\1",
                use_regex=True,
                in_place=True,
            )
            with open(path, encoding="utf-8") as f:
                assert f.read() == "15/01/2024"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_invalid_regex_returns_error(self) -> None:
        path = _write_tmp("some text")
        try:
            result = await search_replace_text(
                resource_uri=path,
                pattern="[invalid",
                replacement="x",
                use_regex=True,
                in_place=True,
            )
            assert "Regex error" in result["text"]
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_regex_max_replacements(self) -> None:
        path = _write_tmp("cat bat sat mat")
        try:
            await search_replace_text(
                resource_uri=path,
                pattern=r"\w+at",
                replacement="X",
                use_regex=True,
                max_replacements=2,
                in_place=True,
            )
            with open(path, encoding="utf-8") as f:
                content = f.read()
            assert content.count("X") == 2
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_missing_file_returns_error(self) -> None:
        result = await search_replace_text(
            resource_uri="/nonexistent/path/file.txt",
            pattern="x",
            replacement="y",
            in_place=True,
        )
        assert result["type"] == "text"
        assert "Error" in result["text"]

    @pytest.mark.asyncio
    async def test_result_has_execution_metadata(self) -> None:
        path = _write_tmp("hello")
        try:
            result = await search_replace_text(
                resource_uri=path,
                pattern="hello",
                replacement="world",
                in_place=True,
            )
            # Decorator injects execution_id and new_tool_call_id on success
            assert "execution_id" in result
            assert "new_tool_call_id" in result
        finally:
            os.unlink(path)
