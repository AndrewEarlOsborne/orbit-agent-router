"""Test suite for Launchpad masking and summarization"""

import pytest
from orbit import DefaultLaunchpad, Launchpad, StationCache, Shuttle
from orbit.protocols import MCPToolResult


class TestDefaultLaunchpad:
    """Test DefaultLaunchpad masking behavior"""

    @pytest.mark.asyncio
    async def test_masking_threshold_default(self) -> None:
        """Test default 2048 character threshold"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station)

        assert launchpad.threshold == 2048

    @pytest.mark.asyncio
    async def test_masking_threshold_custom(self) -> None:
        """Test custom threshold"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station, threshold=100)

        assert launchpad.threshold == 100

    @pytest.mark.asyncio
    async def test_intercept_text_content_under_threshold(self) -> None:
        """Test that content under threshold is not masked"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station, threshold=100)

        result = {"content": [{"type": "text", "text": "short text"}]}

        intercepted = await launchpad._process_result("id-1", "test_tool", result)

        assert intercepted["content"][0]["text"] == "short text"
        assert isinstance(intercepted["content"][0]["text"], str)

    @pytest.mark.asyncio
    async def test_intercept_text_content_over_threshold(self) -> None:
        """Test that content over threshold is masked"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station, threshold=50)

        long_text = "x" * 100
        result = {"content": [{"type": "text", "text": long_text}]}

        intercepted = await launchpad._process_result("id-1", "test_tool", result)

        masked_text = intercepted["content"][0]["text"]
        assert isinstance(masked_text, dict)
        assert masked_text["original_type"] == "string"
        assert masked_text["length"] == 100
        assert "summary" in masked_text
        assert "preview" in masked_text

    @pytest.mark.asyncio
    async def test_intercept_stores_original(self) -> None:
        """Test that original payload is stored before masking"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station, threshold=50)

        long_text = "x" * 100
        result = {"content": [{"type": "text", "text": long_text}]}

        await launchpad._process_result("id-1", "test_tool", result)

        # Shuttle should be docked at station with full payload
        stored = await station.get_payload("id-1")
        assert isinstance(stored, Shuttle)
        assert stored.tool_call_id == "id-1"
        assert stored.tool_name == "test_tool"
        assert stored.payload["content"][0]["text"] == long_text

    @pytest.mark.asyncio
    async def test_mask_nested_dict(self) -> None:
        """Test masking nested dictionary values"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station, threshold=50)

        long_text = "x" * 100
        result = {
            "content": [
                {"type": "data", "data": {"nested": {"long_value": long_text, "short_value": "ok"}}}
            ]
        }

        intercepted = await launchpad._process_result("id-1", "test_tool", result)

        nested_data = intercepted["content"][0]["data"]["nested"]
        assert isinstance(nested_data["long_value"], dict)
        assert nested_data["long_value"]["length"] == 100
        assert nested_data["short_value"] == "ok"

    @pytest.mark.asyncio
    async def test_mask_list_items(self) -> None:
        """Test masking items in lists"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station, threshold=50)

        long_text = "x" * 100
        result = {
            "content": [{"type": "data", "data": {"items": ["short", long_text, "another short"]}}]
        }

        intercepted = await launchpad._process_result("id-1", "test_tool", result)

        items = intercepted["content"][0]["data"]["items"]
        assert items[0] == "short"
        assert isinstance(items[1], dict)
        assert items[1]["length"] == 100
        assert items[2] == "another short"

    @pytest.mark.asyncio
    async def test_multiple_content_items(self) -> None:
        """Test handling multiple content items"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station, threshold=50)

        long_text = "x" * 100
        result = {
            "content": [
                {"type": "text", "text": "short"},
                {"type": "text", "text": long_text},
                {"type": "data", "data": {"key": "value"}},
            ]
        }

        intercepted = await launchpad._process_result("id-1", "test_tool", result)

        assert intercepted["content"][0]["text"] == "short"
        assert isinstance(intercepted["content"][1]["text"], dict)
        assert intercepted["content"][2]["data"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_validate_result_with_dict(self) -> None:
        """Test result validation with dict format"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station)

        valid_result = {"content": []}
        assert launchpad._validate_result(valid_result) is True

        invalid_result = {"data": "no content key"}
        assert launchpad._validate_result(invalid_result) is False

    @pytest.mark.asyncio
    async def test_validate_result_with_object(self) -> None:
        """Test result validation with object format"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station)

        valid_result = MCPToolResult(content=[{"type": "text", "text": "test"}])
        assert launchpad._validate_result(valid_result) is True

        class InvalidResult:
            pass

        invalid_result = InvalidResult()
        assert launchpad._validate_result(invalid_result) is False

    @pytest.mark.asyncio
    async def test_invalid_result_raises_error(self) -> None:
        """Test that invalid result raises ValueError"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station)

        invalid_result = {"no_content": "invalid"}

        with pytest.raises(ValueError, match="Invalid tool result"):
            await launchpad._process_result("id-1", "test_tool", invalid_result)

    @pytest.mark.asyncio
    async def test_extract_content_from_dict(self) -> None:
        """Test content extraction from dict"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station)

        result = {"content": [{"type": "text", "text": "test"}]}
        content = launchpad._extract_content(result)

        assert content == [{"type": "text", "text": "test"}]

    @pytest.mark.asyncio
    async def test_extract_content_from_object(self) -> None:
        """Test content extraction from object"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station)

        result = MCPToolResult(content=[{"type": "text", "text": "test"}])
        content = launchpad._extract_content(result)

        assert content == [{"type": "text", "text": "test"}]

    @pytest.mark.asyncio
    async def test_replace_content_in_dict(self) -> None:
        """Test content replacement in dict"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station)

        result = {"content": [{"type": "text", "text": "old"}]}
        new_content = [{"type": "text", "text": "new"}]

        modified = launchpad._replace_content(result, new_content)

        assert modified["content"] == new_content

    @pytest.mark.asyncio
    async def test_replace_content_in_object(self) -> None:
        """Test content replacement in object"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station)

        result = MCPToolResult(content=[{"type": "text", "text": "old"}])
        new_content = [{"type": "text", "text": "new"}]

        modified = launchpad._replace_content(result, new_content)

        assert modified.content == new_content

    @pytest.mark.asyncio
    async def test_preview_truncation(self) -> None:
        """Test that preview is truncated to 100 chars"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station, threshold=50)

        long_text = "a" * 200
        result = {"content": [{"type": "text", "text": long_text}]}

        intercepted = await launchpad._process_result("id-1", "test_tool", result)

        masked_text = intercepted["content"][0]["text"]
        preview = masked_text["preview"]

        assert len(preview) == 103  # 100 chars + "..."
        assert preview.startswith("a" * 100)
        assert preview.endswith("...")


class TestCustomLaunchpad:
    """Test custom Launchpad implementations"""

    @pytest.mark.asyncio
    async def test_custom_summary_logic(self) -> None:
        """Test custom summarization logic"""

        class CustomLaunchpad(Launchpad):
            """Custom launchpad that replaces all text with placeholder"""

            def _generate_summary(self, tool_call_id, tool_name, content):
                summary_content = []
                for item in content:
                    if item.get("type") == "text":
                        summary_content.append(
                            {"type": "text", "text": f"[REDACTED by {tool_name}]"}
                        )
                    else:
                        summary_content.append(item)
                return summary_content

        station = StationCache()
        launchpad = CustomLaunchpad(station=station)

        result = {"content": [{"type": "text", "text": "sensitive data"}]}
        intercepted = await launchpad._process_result("id-1", "test_tool", result)

        assert intercepted["content"][0]["text"] == "[REDACTED by test_tool]"

        # Shuttle should still be docked with original payload
        stored = await station.get_payload("id-1")
        assert isinstance(stored, Shuttle)
        assert stored.payload["content"][0]["text"] == "sensitive data"
