"""
Orbit - Client-side tool data payload handling and masking for agentic systems

Orbit provides observable and deterministic data handling from tool calls,
with robust customizability, masking for sensitive information, and data routing.
"""

from orbit.launchpad import Launchpad, DefaultLaunchpad
from orbit.station import Station, StationCache, StationDB
from orbit.protocols import (
    ToolProtocol,
    MCPToolProtocol,
    LangChainToolProtocol,
    MCPToolResult,
    TextContent,
    DataContent,
    ResourceContent,
)
from orbit.wrappers.mcp_wrapper import (
    wrap_mcp_tool,
    MCPClientInterceptor,
    intercept_mcp_session,
)
from orbit.wrappers.langchain_wrapper import (
    wrap_langchain_tool,
    LangChainToolNodeInterceptor,
    intercept_tool_node,
)

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "Launchpad",
    "DefaultLaunchpad",
    "Station",
    "StationCache",
    "StationDB",
    # Type protocols
    "ToolProtocol",
    "MCPToolProtocol",
    "LangChainToolProtocol",
    "MCPToolResult",
    "TextContent",
    "DataContent",
    "ResourceContent",
    # MCP wrappers
    "wrap_mcp_tool",
    "MCPClientInterceptor",
    "intercept_mcp_session",
    # LangChain wrappers
    "wrap_langchain_tool",
    "LangChainToolNodeInterceptor",
    "intercept_tool_node",
]
