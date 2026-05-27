"""Test suite for MCP tool wrapping functionality"""

import pytest
from typing import Any, Dict, List
from orbit import DefaultLaunchpad, StationCache, Shuttle
from orbit.wrappers.mcp_wrapper import MCPToolWrapper, MCPClientInterceptor
from mcp.types import CallToolResult, ContentBlock, TextContent


class MockMCPTool:
    """Mock MCP tool for testing"""

    def __init__(self, name: str, description: str, result_content: List[Dict[str, Any]]) -> None:
        self.name = name
        self.description = description
        self.inputSchema = {"type": "object", "properties": {}}
        self._result_content = result_content

    async def __call__(self, **kwargs: Any) -> Dict[str, Any]:
        """Return mock result"""
        return {"content": self._result_content}


class MockMCPSession:
    """Mock MCP session for testing"""

    def __init__(self, result_content: List[Dict[str, Any]]) -> None:
        self._result_content = result_content

    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> CallToolResult:
        """Mock call_tool method that returns MCP CallToolResult-like object"""

        content_objects: List[ContentBlock] = []
        for item in self._result_content:
            if isinstance(item, dict):
                item_type = item.get("type", "text")
                if item_type == "text":
                    content_objects.append(TextContent(type="text", text=item.get("text", "")))
            else:
                content_objects.append(item)

        return CallToolResult(content=content_objects)


class TestMCPToolWrapper:
    """Test MCP tool wrapping"""

    @pytest.mark.asyncio
    async def test_wrapper_preserves_tool_metadata(self) -> None:
        """Test that wrapper preserves tool name, description, and schema"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station)

        original_tool = MockMCPTool(
            name="test_tool",
            description="A test tool",
            result_content=[{"type": "text", "text": "short result"}],
        )

        wrapped_tool = MCPToolWrapper(original_tool, launchpad)

        assert wrapped_tool.name == "test_tool"
        assert wrapped_tool.description == "A test tool"
        assert wrapped_tool.inputSchema == {"type": "object", "properties": {}}

    @pytest.mark.asyncio
    async def test_wrapper_docks_shuttle_at_station(self) -> None:
        """Test that wrapper docks a shuttle carrying the full payload at the station"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station)

        result_content = [{"type": "text", "text": "test result"}]
        original_tool = MockMCPTool(
            name="test_tool", description="A test tool", result_content=result_content
        )

        wrapped_tool = MCPToolWrapper(original_tool, launchpad)
        result = await wrapped_tool(arg="value")
        assert result["content"][0]["text"] == "test result"

        # Station should have one docked shuttle
        assert len(station._cache) > 0

        # Shuttle at station carries the full payload
        tool_call_ids = list(station._cache.keys())
        docked_shuttle = await station.get_payload(tool_call_ids[0])

        assert isinstance(docked_shuttle, Shuttle)
        assert docked_shuttle.payload == {"content": result_content}

    @pytest.mark.asyncio
    async def test_wrapper_masks_long_content(self) -> None:
        """Test that wrapper masks content exceeding threshold"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station, threshold=100)

        text_length = 1000

        long_text = "x" * text_length
        result_content = [{"type": "text", "text": long_text}]

        original_tool = MockMCPTool(
            name="test_tool", description="A test tool", result_content=result_content
        )

        wrapped_tool = MCPToolWrapper(original_tool, launchpad)
        result = await wrapped_tool(arg="value")

        # Result should be masked
        assert result["content"][0]["type"] == "text"
        assert isinstance(result["content"][0]["text"], dict)
        assert result["content"][0]["text"]["original_type"] == "string"
        assert result["content"][0]["text"]["length"] == text_length
        assert "summary" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_wrapper_preserves_short_content(self) -> None:
        """Test that wrapper preserves content under threshold"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station, threshold=100)

        short_text = "short"
        result_content = [{"type": "text", "text": short_text}]

        original_tool = MockMCPTool(
            name="test_tool", description="A test tool", result_content=result_content
        )

        wrapped_tool = MCPToolWrapper(original_tool, launchpad)
        result = await wrapped_tool(arg="value")

        # Result should not be masked
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == short_text

    @pytest.mark.asyncio
    async def test_wrapper_handles_multiple_content_items(self) -> None:
        """Test that wrapper handles multiple content items"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station, threshold=50)

        text_length = 1000

        result_content: List[Dict[str, Any]] = [
            {"type": "text", "text": "short"},
            {"type": "text", "text": "x" * text_length},  # Long, should be masked
            {"type": "data", "data": {"key": "value"}},
        ]

        original_tool = MockMCPTool(
            name="test_tool", description="A test tool", result_content=result_content
        )

        wrapped_tool = MCPToolWrapper(original_tool, launchpad)
        result = await wrapped_tool(arg="value")

        # First item should not be masked
        assert result["content"][0]["text"] == "short"

        # Second item should be masked
        assert isinstance(result["content"][1]["text"], dict)
        assert result["content"][1]["text"]["length"] == text_length

        # Third item should not be masked (data type)
        assert result["content"][2]["data"] == {"key": "value"}


class TestMCPClientInterceptor:
    """Test MCP session interception"""

    @pytest.mark.asyncio
    async def test_session_interceptor_enables(self) -> None:
        """Test that session interceptor can be enabled"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station)

        result_content = [{"type": "text", "text": "result"}]
        session = MockMCPSession(result_content)

        original_call_tool = session.call_tool
        interceptor = MCPClientInterceptor(session, launchpad)
        interceptor.enable()

        # call_tool should be replaced
        assert session.call_tool != original_call_tool

    @pytest.mark.asyncio
    async def test_session_interceptor_disables(self) -> None:
        """Test that session interceptor can be disabled"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station)

        result_content = [{"type": "text", "text": "result"}]
        session = MockMCPSession(result_content)

        original_call_tool = session.call_tool
        interceptor = MCPClientInterceptor(session, launchpad)
        interceptor.enable()
        interceptor.disable()

        # call_tool should be restored
        assert session.call_tool == original_call_tool

    @pytest.mark.asyncio
    async def test_session_interceptor_stores_payload(self) -> None:
        """Test that session interceptor docks full payload at station"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station)

        result_content = [{"type": "text", "text": "result"}]
        session = MockMCPSession(result_content)

        interceptor = MCPClientInterceptor(session, launchpad)
        interceptor.enable()

        result = await session.call_tool("test_tool", {"arg": "value"})

        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "result"

        # Station should have one docked shuttle
        assert station._cache is not None
        assert len(station._cache.keys()) == 1

        # Shuttle at station carries the full payload
        tool_call_ids = list(station._cache.keys())
        docked_shuttle = await station.get_payload(tool_call_ids[0])

        assert isinstance(docked_shuttle, Shuttle)
        assert docked_shuttle.payload == {"content": result_content}

    @pytest.mark.asyncio
    async def test_session_interceptor_masks_content(self) -> None:
        """Test that session interceptor masks long content"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station, threshold=50)

        long_text = "x" * 1000
        result_content = [{"type": "text", "text": long_text}]
        session = MockMCPSession(result_content)

        interceptor = MCPClientInterceptor(session, launchpad)
        interceptor.enable()

        result = await session.call_tool("test_tool", {"arg": "value"})

        # Result should be masked
        assert isinstance(result.content, list)
        assert isinstance(result.content[0], TextContent)
        assert "100" in result.content[0].text
