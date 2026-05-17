"""Type definitions and protocols for Orbit"""

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class ToolProtocol(Protocol):
    """Protocol for tool objects that can be wrapped by Orbit"""

    name: str
    description: str


@runtime_checkable
class MCPToolProtocol(Protocol):
    """Protocol for MCP tool objects"""

    name: str
    description: str
    inputSchema: Dict[str, Any]


@runtime_checkable
class LangChainToolProtocol(Protocol):
    """Protocol for LangChain tool objects"""

    name: str
    description: str
    args_schema: Any


class ToolResult(Protocol):
    """Protocol for tool execution results"""

    def __init__(self, content: List[Dict[str, Any]]) -> None: ...


class MCPToolResult:
    """MCP tool result structure"""

    def __init__(self, content: List[Dict[str, Any]]) -> None:
        self.content = content

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {"content": self.content}


class ContentItem:
    """Base class for content items in tool results"""

    def __init__(self, type: str, **kwargs: Any) -> None:
        self.type = type
        self.data = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {"type": self.type, **self.data}


class TextContent(ContentItem):
    """Text content item"""

    def __init__(self, text: str) -> None:
        super().__init__("text", text=text)
        self.text = text


class DataContent(ContentItem):
    """Data content item"""

    def __init__(self, data: Dict[str, Any]) -> None:
        super().__init__("data", data=data)
        self.data_payload = data


class ResourceContent(ContentItem):
    """Resource content item"""

    def __init__(self, uri: str, mimeType: str) -> None:
        super().__init__("resource", uri=uri, mimeType=mimeType)
        self.uri = uri
        self.mimeType = mimeType
