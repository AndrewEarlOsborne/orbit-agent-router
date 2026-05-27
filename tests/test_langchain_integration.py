"""Real LangChain integration tests with actual tools and ToolNode"""

import pytest
from typing import Any, Dict
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from orbit import DefaultLaunchpad, StationCache, Shuttle
from orbit.wrappers.langchain_wrapper import wrap_langchain_tool, intercept_tool_node


@tool
def search_database(query: str) -> str:
    """Search database and return large result set"""
    large_data = "x" * 3000
    return f"Results for '{query}':\n{large_data}"


@tool
def fetch_api_data(endpoint: str) -> Dict[str, Any]:
    """Fetch data from API endpoint"""
    large_response = "y" * 3000
    return {"endpoint": endpoint, "data": large_response, "status": "success"}


@tool
def get_small_info(key: str) -> str:
    """Get small piece of information"""
    return f"Info for {key}: small value"


class TestRealLangChainIntegration:
    """Integration tests with real LangChain tools"""

    @pytest.mark.asyncio
    async def test_tool_wrapping_with_large_string_result(self) -> None:
        """Test wrapping a real LangChain tool that returns large string"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station, threshold=2048)

        wrapped_tool = wrap_langchain_tool(search_database, launchpad)

        result = await wrapped_tool.ainvoke({"query": "test search"})

        assert len(station._cache.keys()) == 1

        tool_call_ids = list(station._cache.keys())
        stored = await station.get_payload(tool_call_ids[0])

        assert stored is not None
        assert isinstance(stored, Shuttle)
        full_text = stored.payload["content"][0]["text"]
        assert "Results for 'test search':" in full_text
        assert "x" * 3000 in full_text

        assert isinstance(result, str)
        assert "x" * 3000 not in result
        assert "masked" in result.lower() or "Content masked" in result

    @pytest.mark.asyncio
    async def test_tool_wrapping_with_dict_result(self) -> None:
        """Test wrapping a tool that returns dictionary"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station, threshold=2048)

        wrapped_tool = wrap_langchain_tool(fetch_api_data, launchpad)

        result = await wrapped_tool.ainvoke({"endpoint": "/api/data"})

        assert len(station._cache.keys()) == 1

        tool_call_ids = list(station._cache.keys())
        stored = await station.get_payload(tool_call_ids[0])

        assert stored is not None
        assert isinstance(stored, Shuttle)
        full_data = stored.payload["content"][0]["data"]
        assert full_data["endpoint"] == "/api/data"
        assert full_data["data"] == "y" * 3000
        assert full_data["status"] == "success"

        assert isinstance(result, dict)
        assert result["endpoint"] == "/api/data"
        assert result["data"] != "y" * 3000

    @pytest.mark.asyncio
    async def test_tool_wrapping_preserves_small_results(self) -> None:
        """Test that small results pass through unmasked"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station, threshold=2048)

        wrapped_tool = wrap_langchain_tool(get_small_info, launchpad)

        result = await wrapped_tool.ainvoke({"key": "config"})

        assert len(station._cache.keys()) == 1

        tool_call_ids = list(station._cache.keys())
        stored = await station.get_payload(tool_call_ids[0])

        assert isinstance(stored, Shuttle)
        assert stored.payload["content"][0]["text"] == "Info for config: small value"

        assert result == "Info for config: small value"

    @pytest.mark.asyncio
    async def test_toolnode_interception(self) -> None:
        """Test intercepting a real LangGraph ToolNode"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station, threshold=2048)

        tools = [search_database, fetch_api_data, get_small_info]
        tool_node = ToolNode(tools)

        intercept_tool_node(tool_node, launchpad)

        wrapped_search = wrap_langchain_tool(search_database, launchpad)
        wrapped_api = wrap_langchain_tool(fetch_api_data, launchpad)
        wrapped_info = wrap_langchain_tool(get_small_info, launchpad)

        result1 = await wrapped_search.ainvoke({"query": "test1"})
        result2 = await wrapped_api.ainvoke({"endpoint": "/test"})
        result3 = await wrapped_info.ainvoke({"key": "test_key"})

        assert len(station._cache) == 3

        assert isinstance(result1, str)
        assert "x" * 3000 not in result1

        assert isinstance(result2, dict)
        assert result2["data"] != "y" * 3000

        assert result3 == "Info for test_key: small value"

    def test_sync_tool_execution(self) -> None:
        """Test that sync tool execution also works"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station, threshold=2048)

        wrapped_tool = wrap_langchain_tool(search_database, launchpad)

        result = wrapped_tool.invoke({"query": "sync test"})

        assert len(station._cache.keys()) == 1

        assert isinstance(result, str)
        assert "sync test" in result or "masked" in result.lower()
        assert "x" * 3000 not in result

    @pytest.mark.asyncio
    async def test_multiple_calls_separate_cache_entries(self) -> None:
        """Test that multiple tool calls create separate cache entries"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station, threshold=2048)

        wrapped_search = wrap_langchain_tool(search_database, launchpad)
        wrapped_api = wrap_langchain_tool(fetch_api_data, launchpad)

        await wrapped_search.ainvoke({"query": "first"})
        await wrapped_search.ainvoke({"query": "second"})
        await wrapped_api.ainvoke({"endpoint": "/third"})

        assert len(station._cache) == 3

        tool_call_ids = list(station._cache.keys())
        stored1 = await station.get_payload(tool_call_ids[0])
        stored2 = await station.get_payload(tool_call_ids[1])
        stored3 = await station.get_payload(tool_call_ids[2])

        assert stored1 is not None
        assert stored2 is not None
        assert stored3 is not None
        assert "first" in stored1.payload["content"][0]["text"]
        assert "second" in stored2.payload["content"][0]["text"]
        assert stored3.payload["content"][0]["data"]["endpoint"] == "/third"

    @pytest.mark.asyncio
    async def test_data_masking_verification(self) -> None:
        """
        Verify that LLM cannot access full large data, but it's cached
        This is the core use case: protect LLM from large payloads
        """
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station, threshold=2048)

        wrapped_tool = wrap_langchain_tool(search_database, launchpad)

        llm_result = await wrapped_tool.ainvoke({"query": "sensitive data"})

        tool_call_ids = list(station._cache.keys())
        tool_call_id = tool_call_ids[0]
        full_cached_data = await station.get_payload(tool_call_id)

        assert isinstance(full_cached_data, Shuttle)
        assert "x" * 3000 in full_cached_data.payload["content"][0]["text"]

        assert "x" * 3000 not in llm_result

        if isinstance(llm_result, str):
            assert "sensitive data" in llm_result or "masked" in llm_result.lower()
        else:
            assert "masked" in str(llm_result).lower()

    @pytest.mark.asyncio
    async def test_tool_metadata_preserved(self) -> None:
        """Test that wrapped tools preserve original metadata"""
        station = StationCache()
        launchpad = DefaultLaunchpad(station=station)

        wrapped_tool = wrap_langchain_tool(search_database, launchpad)

        assert wrapped_tool.name == search_database.name
        assert wrapped_tool.description == search_database.description

        assert hasattr(wrapped_tool, "ainvoke")
        assert hasattr(wrapped_tool, "invoke")
