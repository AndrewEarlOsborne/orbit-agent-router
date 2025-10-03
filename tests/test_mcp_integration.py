"""Real MCP integration tests with actual server and client"""

import pytest
import sys
from pathlib import Path
from mcp import types
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

from orbit import DefaultLaunchpad, StationCache
from orbit.wrappers.mcp_wrapper import intercept_mcp_session
from orbit.wrappers.mcp_wrapper import MCPClientInterceptor


@pytest.fixture
async def mcp_server_script(tmp_path: Path) -> Path:
    """Create a real MCP server script that returns large data"""
    server_code = '''
import asyncio
import json
from mcp.server.stdio import stdio_server
from mcp.server import Server
from mcp import types

server = Server("test-server")

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="fetch_large_data",
            description="Fetches a large dataset for testing",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }
        ),
        types.Tool(
            name="fetch_small_data",
            description="Fetches a small dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Execute a tool"""
    if name == "fetch_large_data":
        large_text = "x" * 3000
        return [
            types.TextContent(
                type="text",
                text=f"Query: {arguments.get('query', '')}"
            ),
            types.TextContent(
                type="text",
                text=large_text
            )
        ]
    elif name == "fetch_small_data":
        return [
            types.TextContent(
                type="text",
                text=f"Small result for: {arguments.get('query', '')}"
            )
        ]
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
'''

    server_file = tmp_path / "test_server.py"
    server_file.write_text(server_code)
    return server_file


class TestRealMCPIntegration:
    """Integration tests with real MCP server and client"""

    @pytest.mark.asyncio
    async def test_end_to_end_with_real_mcp_server(self, mcp_server_script: Path) -> None:
        """Test complete workflow with real MCP server/client"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station, threshold=2048)

        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(mcp_server_script)],
            env=None
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                intercept_mcp_session(session, launchpad)

                result = await session.call_tool("fetch_large_data", {"query": "test query"})

                assert station._cache is not None
                assert len(station._cache.keys()) == 1

                tool_call_ids = list(station._cache.keys())
                stored = await station.get_payload(tool_call_ids[0])

                assert stored is not None
                assert isinstance(stored, dict)
                assert len(stored["content"]) == 2
                assert stored["content"][0]["text"] == "Query: test query"
                assert len(stored["content"][1]["text"]) == 3000

                assert isinstance(result, types.CallToolResult)
                assert result.content[0].text == "Query: test query"

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_with_real_server(self, mcp_server_script: Path) -> None:
        """Test multiple tool calls are tracked separately"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station, threshold=2048)

        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(mcp_server_script)],
            env=None
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                intercept_mcp_session(session, launchpad)

                await session.call_tool("fetch_large_data", {"query": "first"})
                await session.call_tool("fetch_small_data", {"query": "second"})

                assert len(station._cache) == 2

                tool_call_ids = list(station._cache.keys())
                stored1 = await station.get_payload(tool_call_ids[0])
                stored2 = await station.get_payload(tool_call_ids[1])

                assert stored1["content"][0]["text"] == "Query: first"
                assert stored2["content"][0]["text"] == "Small result for: second"

    @pytest.mark.asyncio
    async def test_interceptor_disable_with_real_server(self, mcp_server_script: Path) -> None:
        """Test that disabling interceptor restores original behavior"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station, threshold=2048)

        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(mcp_server_script)],
            env=None
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                interceptor = intercept_mcp_session(session, launchpad)

                await session.call_tool("fetch_large_data", {"query": "test1"})
                assert station._cache is not None
                assert len(station._cache.keys()) == 1

                interceptor.disable()

                result2 = await session.call_tool("fetch_large_data", {"query": "test2"})

                assert len(station._cache.keys()) == 1
                assert len(result2.content[1].text) == 3000

    @pytest.mark.asyncio
    async def test_data_not_visible_to_llm_but_retrievable(self, mcp_server_script: Path) -> None:
        """
        Test that large data is masked from LLM but retrievable from cache
        This simulates the key use case: LLM sees summary, developer can retrieve full data
        """
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station, threshold=2048)

        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(mcp_server_script)],
            env=None
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                interceptor = MCPClientInterceptor(session, launchpad)
                interceptor.enable()

                result = await session.call_tool("fetch_large_data", {"query": "sensitive query"})

                tool_call_ids = list(station._cache.keys())
                tool_call_id = tool_call_ids[0]

                stored_full_data = await station.get_payload(tool_call_id)

                assert isinstance(result, types.CallToolResult)
                llm_visible_content = result.content[1]

                if isinstance(stored_full_data, dict):
                    assert stored_full_data["content"][1]["text"] == "x" * 3000
                else:
                    assert stored_full_data.content[1].text == "x" * 3000

                assert isinstance(llm_visible_content.text, (dict, str))
                if isinstance(llm_visible_content.text, str):
                    assert llm_visible_content.text != "x" * 3000
                else:
                    assert "masked" in str(llm_visible_content.text).lower() or "length" in llm_visible_content.text
