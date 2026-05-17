"""Launchpad classes for intercepting and summarizing tool payloads"""

from typing import Any, Dict, List, Union
from abc import ABC, abstractmethod
import logging
import copy

from orbit.station import Station, StationCache
from orbit.protocols import MCPToolProtocol, LangChainToolProtocol, ToolProtocol

logger = logging.getLogger(__name__)


class Launchpad(ABC):
    """Base class for payload interceptors and summarizers"""

    def __init__(self, station: Union[Station, None] = None) -> None:
        """
        Initialize Launchpad with a storage backend

        Args:
            station: Storage backend (StationCache or StationDB) for full payloads.
                     If None, creates a default StationCache.
        """
        self.station = station if station is not None else StationCache()
        logger.info("Launchpad initialized with station: %s", type(self.station).__name__)

    def stage(self, tool: ToolProtocol) -> ToolProtocol:
        """
        Wrap a tool to enable payload interception

        Args:
            tool: Tool object (MCP tool, LangChain tool, etc.)

        Returns:
            Wrapped tool with identical signature but modified execution behavior

        Raises:
            TypeError: If tool type is not supported
        """
        if isinstance(tool, MCPToolProtocol):
            logger.debug("Staging MCP tool: %s", tool.name)
            from orbit.wrappers.mcp_wrapper import wrap_mcp_tool

            return wrap_mcp_tool(tool, self)
        elif isinstance(tool, LangChainToolProtocol):
            logger.debug("Staging LangChain tool: %s", tool.name)
            from orbit.wrappers.langchain_wrapper import wrap_langchain_tool

            return wrap_langchain_tool(tool, self)
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
        Generate summary/masked version of tool result content

        This method should be overridden in custom Launchpad implementations
        to provide domain-specific summarization logic.

        Args:
            tool_call_id: Unique identifier for this tool call
            tool_name: Name of the tool that was called
            content: List of content objects from tool result

        Returns:
            List of summarized/masked content objects
        """
        pass

    async def _process_result(self, tool_call_id: str, tool_name: str, result: Any) -> Any:
        """
        Internal method to intercept and process tool results

        Args:
            tool_call_id: Unique identifier for this tool call
            tool_name: Name of the tool that was called
            result: Raw result from tool execution

        Returns:
            Modified result with summarized content

        Raises:
            ValueError: If result is not a valid tool message
        """
        logger.debug(
            "Intercepting result for tool_call_id: %s, tool_name: %s", tool_call_id, tool_name
        )

        if not self._validate_result(result):
            raise ValueError(f"Invalid tool result for tool_call_id: {tool_call_id}")

        original_result = copy.deepcopy(result)
        await self.station.store_payload(tool_call_id, original_result)
        logger.debug("Stored original payload for tool_call_id: %s", tool_call_id)

        content = self._extract_content(result)

        summary_content = self._generate_summary(tool_call_id, tool_name, content)
        logger.debug("Generated summary for tool_call_id: %s", tool_call_id)

        modified_result = self._replace_content(result, summary_content)

        return modified_result

    def _validate_result(self, result: Any) -> bool:
        """
        Validate that result is a valid tool message

        Args:
            result: Result to validate

        Returns:
            True if valid, False otherwise
        """
        if result is None:
            return False

        if isinstance(result, dict):
            return "content" in result

        if hasattr(result, "content"):
            return True

        return False

    def _extract_content(self, result: Any) -> List[Dict[str, Any]]:
        """
        Extract content list from result

        Args:
            result: Tool result

        Returns:
            List of content dictionaries
        """
        if isinstance(result, dict):
            content = result.get("content", [])
        elif hasattr(result, "content"):
            content = result.content
        else:
            content = []

        if not isinstance(content, list):
            content = [content]

        return content

    def _replace_content(self, result: Any, new_content: List[Dict[str, Any]]) -> Any:
        """
        Replace content in result with new content

        Args:
            result: Original result
            new_content: New content to insert

        Returns:
            Result with replaced content
        """
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
        """
        Initialize DefaultLaunchpad

        Args:
            station: Storage backend for full payloads
            threshold: Character threshold for masking (default: 2048)
        """
        super().__init__(station)
        self.threshold = threshold
        logger.info("DefaultLaunchpad initialized with threshold: %d", threshold)

    def _generate_summary(
        self, tool_call_id: str, tool_name: str, content: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Default summary: mask any string values exceeding threshold chars

        Returns metadata about masked content including type and length.

        Args:
            tool_call_id: Unique identifier for this tool call
            tool_name: Name of the tool that was called
            content: List of content objects from tool result

        Returns:
            List of summarized/masked content objects
        """
        summary_content: List[Dict[str, Any]] = []

        for item in content:
            if not isinstance(item, dict):
                summary_content.append(item)
                continue

            item_type = item.get("type", "unknown")
            masked_item = self._mask_item(item, item_type)
            summary_content.append(masked_item)

        return summary_content

    def _mask_item(self, item: Dict[str, Any], item_type: str) -> Dict[str, Any]:
        """
        Mask a single content item if it exceeds threshold

        Args:
            item: Content item to potentially mask
            item_type: Type of the content item

        Returns:
            Masked or original item
        """
        masked_item = copy.copy(item)

        for key, value in item.items():
            if key == "type":
                continue

            if isinstance(value, str) and len(value) > self.threshold:
                masked_item[key] = {
                    "original_type": "string",
                    "length": len(value),
                    "summary": f"Content masked - exceeds {self.threshold} chars",
                    "preview": value[: min(100, len(value))] + "...",
                }
                logger.debug(
                    "Masked field '%s' in item type '%s' (length: %d)", key, item_type, len(value)
                )

            elif isinstance(value, dict):
                masked_item[key] = self._mask_dict(value)

            elif isinstance(value, list):
                masked_item[key] = self._mask_list(value)

        return masked_item

    def _mask_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively mask dictionary values

        Args:
            data: Dictionary to mask

        Returns:
            Masked dictionary
        """
        masked = {}
        for key, value in data.items():
            if isinstance(value, str) and len(value) > self.threshold:
                masked[key] = {
                    "original_type": "string",
                    "length": len(value),
                    "summary": f"Content masked - exceeds {self.threshold} chars",
                    "preview": value[: min(100, len(value))] + "...",
                }
            elif isinstance(value, dict):
                masked[key] = self._mask_dict(value)
            elif isinstance(value, list):
                masked[key] = self._mask_list(value)
            else:
                masked[key] = value
        return masked

    def _mask_list(self, data: List[Any]) -> List[Any]:
        """
        Recursively mask list items

        Args:
            data: List to mask

        Returns:
            Masked list
        """
        masked = []
        for item in data:
            if isinstance(item, str) and len(item) > self.threshold:
                masked.append(
                    {
                        "original_type": "string",
                        "length": len(item),
                        "summary": f"Content masked - exceeds {self.threshold} chars",
                        "preview": item[: min(100, len(item))] + "...",
                    }
                )
            elif isinstance(item, dict):
                masked.append(self._mask_dict(item))
            elif isinstance(item, list):
                masked.append(self._mask_list(item))
            else:
                masked.append(item)
        return masked
