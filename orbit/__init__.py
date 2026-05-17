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
from orbit.shuttles.mcp_shuttle import (
    launch_mcp_shuttle,
    MCPClientInterceptor,
    intercept_mcp_session,
)
from orbit.shuttles.langchain_shuttle import (
    launch_langchain_shuttle,
    LangChainToolNodeInterceptor,
    intercept_tool_node,
)
from orbit.transformations import (
    DataType,
    TransformationRegistry,
    orbit_transformation_tool_mcp,
    ResourceManager,
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
    # MCP shuttles
    "launch_mcp_shuttle",
    "MCPClientInterceptor",
    "intercept_mcp_session",
    # LangChain shuttles
    "launch_langchain_shuttle",
    "LangChainToolNodeInterceptor",
    "intercept_tool_node",
    # Transformations framework
    "DataType",
    "TransformationRegistry",
    "orbit_transformation_tool_mcp",
    "ResourceManager",
]
