"""Launchpad classes for wrapping tools and routing payload shuttles to the station"""

from typing import Any, Dict, List, Union, cast
from abc import ABC, abstractmethod
import logging
import copy

from orbit.station import Station, StationCache
from orbit.shuttle import Shuttle
from orbit.protocols import MCPToolProtocol, LangChainToolProtocol, ToolProtocol

logger = logging.getLogger(__name__)


class Launchpad(ABC):
    """Base class for payload interceptors and summarizers"""

    def __init__(self, station: Union[Station, None] = None) -> None:
        self.station = station if station is not None else StationCache()
        logger.info("Launchpad initialized with station: %s", type(self.station).__name__)

    def stage(self, tool: ToolProtocol) -> ToolProtocol:
        """
        Wrap a tool into an Orbit Wrapped Tool

        The wrapped tool has an identical interface to the original but
        intercepts results: the full payload is packaged as a Shuttle and
        stored at the Station; a manifest is returned to the agent.

        Raises:
            TypeError: If tool type is not supported
        """
        if isinstance(tool, MCPToolProtocol):
            logger.debug("Wrapping MCP tool: %s", tool.name)
            from orbit.wrappers.mcp_wrapper import wrap_mcp_tool

            return cast(ToolProtocol, wrap_mcp_tool(tool, self))
        elif isinstance(tool, LangChainToolProtocol):
            logger.debug("Wrapping LangChain tool: %s", tool.name)
            from orbit.wrappers.langchain_wrapper import wrap_langchain_tool

            return cast(ToolProtocol, wrap_langchain_tool(tool, self))
        else:
            raise TypeError(
                f"Unsupported tool type: {type(tool)}. "
                "Must implement MCPToolProtocol or LangChainToolProtocol"
            )

    @abstractmethod
    def _generate_summary(
        self, tool_call_id: str, tool_name: str, content: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate summarized/masked version of tool result content.

        Override in custom Launchpad subclasses to implement domain-specific
        summarization. Receives the full content list; returns the manifest
        content list that will be seen by the agent.
        """
        pass

    async def _process_result(self, tool_call_id: str, tool_name: str, result: Any) -> Any:
        """
        Intercept a tool result: store a Shuttle at the station and return a manifest.

        Raises:
            ValueError: If result is not a valid tool message
        """
        logger.debug(
            "Intercepting result for tool_call_id: %s, tool_name: %s", tool_call_id, tool_name
        )

        if not self._validate_result(result):
            raise ValueError(f"Invalid tool result for tool_call_id: {tool_call_id}")

        original_result = copy.deepcopy(result)
        content = self._extract_content(original_result)

        shuttle = Shuttle(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            payload=original_result,
            content=content,
        )
        await self.station.store_payload(tool_call_id, shuttle)
        logger.debug("Docked shuttle at station for tool_call_id: %s", tool_call_id)

        summary_content = self._generate_summary(tool_call_id, tool_name, content)
        logger.debug("Generated summary for tool_call_id: %s", tool_call_id)

        return self._replace_content(result, summary_content)

    def _validate_result(self, result: Any) -> bool:
        if result is None:
            return False
        if isinstance(result, dict):
            return "content" in result
        return hasattr(result, "content")

    def _extract_content(self, result: Any) -> List[Dict[str, Any]]:
        if isinstance(result, dict):
            content = result.get("content", [])
        elif hasattr(result, "content"):
            content = result.content
        else:
            content = []
        return content if isinstance(content, list) else [content]

    def _replace_content(self, result: Any, new_content: List[Dict[str, Any]]) -> Any:
        if isinstance(result, dict):
            modified = copy.copy(result)
            modified["content"] = new_content
            return modified
        elif hasattr(result, "content"):
            result.content = new_content
            return result
        else:
            return {"content": new_content}


class DefaultLaunchpad(Launchpad):
    """Default implementation with 2048-character masking threshold"""

    def __init__(self, station: Union[Station, None] = None, threshold: int = 2048) -> None:
        super().__init__(station)
        self.threshold = threshold
        logger.info("DefaultLaunchpad initialized with threshold: %d", threshold)

    def _generate_summary(
        self, tool_call_id: str, tool_name: str, content: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Mask any string values exceeding threshold chars; preserve everything else."""
        summary_content: List[Dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                summary_content.append(item)
                continue
            item_type = item.get("type", "unknown")
            summary_content.append(self._mask_item(item, item_type))
        return summary_content

    def _mask_value(self, value: Any) -> Any:
        """Recursively mask a value — returns mask dict, masked container, or original."""
        if isinstance(value, str) and len(value) > self.threshold:
            return {
                "original_type": "string",
                "length": len(value),
                "summary": f"Content masked - exceeds {self.threshold} chars",
                "preview": value[: min(100, len(value))] + "...",
            }
        if isinstance(value, dict):
            return {k: self._mask_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._mask_value(item) for item in value]
        return value

    def _mask_item(self, item: Dict[str, Any], item_type: str) -> Dict[str, Any]:
        """Mask a single MCP content item, preserving the top-level 'type' key."""
        masked_item: Dict[str, Any] = {}
        for key, value in item.items():
            if key == "type":
                masked_item[key] = value
                continue
            masked_val = self._mask_value(value)
            if isinstance(value, str) and len(value) > self.threshold:
                logger.debug(
                    "Masked field '%s' in item type '%s' (length: %d)", key, item_type, len(value)
                )
            masked_item[key] = masked_val
        return masked_item
