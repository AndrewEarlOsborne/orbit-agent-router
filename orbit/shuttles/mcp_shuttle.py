"""MCP shuttle and session interception functionality"""

from typing import TYPE_CHECKING, Any, Dict
import logging
import uuid

if TYPE_CHECKING:
    from orbit.launchpad import Launchpad

logger = logging.getLogger(__name__)


class MCPShuttle:
    """Shuttle for MCP tools that intercepts results and ferries payloads to the station"""

    def __init__(self, original_tool: Any, launchpad: "Launchpad") -> None:
        """
        Initialize MCP shuttle

        Args:
            original_tool: Original MCP tool object
            launchpad: Launchpad instance that launched this shuttle
        """
        self._original_tool = original_tool
        self._launchpad = launchpad

        self.name = original_tool.name
        self.description = original_tool.description
        self.inputSchema = original_tool.inputSchema
        logger.debug("Created MCPShuttle for tool: %s", self.name)

    async def __call__(self, **kwargs: Any) -> Any:
        """
        Execute tool with interception

        Args:
            **kwargs: Tool arguments

        Returns:
            Intercepted and summarized result
        """
        tool_call_id = str(uuid.uuid4())
        logger.debug("Executing MCP shuttle '%s' with tool_call_id: %s", self.name, tool_call_id)

        if hasattr(self._original_tool, "__call__"):
            result = await self._original_tool(**kwargs)
        else:
            raise AttributeError(f"Tool {self.name} is not callable")

        intercepted_result = await self._launchpad._process_result(tool_call_id, self.name, result)

        return intercepted_result


def launch_mcp_shuttle(tool: Any, launchpad: "Launchpad") -> Any:
    """
    Launch an MCP tool as a shuttle, intercepting results before they reach the agent

    The interception occurs after tool execution, before returning
    the result to the agent. The full payload is stored at the station;
    a manifest is returned to the LLM.

    Args:
        tool: MCP tool object to wrap as a shuttle
        launchpad: Launchpad instance that will manage interception

    Returns:
        MCPShuttle wrapping the original tool
    """
    return MCPShuttle(tool, launchpad)


class MCPClientInterceptor:
    """
    Session-level interceptor for MCP clients

    Intercepts calls at the session.call_tool level rather than
    wrapping individual tools as shuttles.
    """

    def __init__(self, session: Any, launchpad: "Launchpad") -> None:
        """
        Initialize session interceptor

        Args:
            session: MCP client session object
            launchpad: Launchpad instance for interception
        """
        self._session = session
        self._launchpad = launchpad
        self._original_call_tool = session.call_tool
        logger.info("MCPClientInterceptor initialized for session")

    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """
        Intercept call_tool invocation

        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool

        Returns:
            Intercepted result with payload stored at station
        """
        tool_call_id = str(uuid.uuid4())
        logger.debug(
            "Intercepting call_tool for tool: %s, tool_call_id: %s", tool_name, tool_call_id
        )

        result = await self._original_call_tool(tool_name, tool_args)

        wrapped_result = self._wrap_mcp_result(result)

        intercepted_result = await self._launchpad._process_result(
            tool_call_id, tool_name, wrapped_result
        )

        unwrapped_result = self._unwrap_mcp_result(intercepted_result, result)

        return unwrapped_result

    def _wrap_mcp_result(self, result: Any) -> Dict[str, Any]:
        """
        Wrap MCP CallToolResult into standard dict format

        Args:
            result: MCP CallToolResult object

        Returns:
            Dict with content as list of dicts
        """
        if hasattr(result, "content"):
            content_dicts = []
            for item in result.content:
                if hasattr(item, "type"):
                    item_dict = {"type": item.type}
                    if hasattr(item, "text"):
                        item_dict["text"] = item.text
                    if hasattr(item, "data"):
                        item_dict["data"] = item.data
                    if hasattr(item, "uri"):
                        item_dict["uri"] = item.uri
                    if hasattr(item, "mimeType"):
                        item_dict["mimeType"] = item.mimeType
                    content_dicts.append(item_dict)
                else:
                    content_dicts.append(item if isinstance(item, dict) else {"data": item})
            return {"content": content_dicts}
        return result

    def _unwrap_mcp_result(self, intercepted_result: Any, original_result: Any) -> Any:
        """
        Unwrap dict result back to MCP CallToolResult format

        Args:
            intercepted_result: Intercepted dict result
            original_result: Original MCP CallToolResult

        Returns:
            MCP CallToolResult with modified content
        """
        if isinstance(intercepted_result, dict) and "content" in intercepted_result:
            if hasattr(original_result, "content"):
                try:
                    from mcp import types

                    new_content = []
                    for item in intercepted_result["content"]:
                        if not isinstance(item, dict):
                            new_content.append(item)
                            continue

                        item_type = item.get("type", "text")
                        try:
                            if item_type == "text":
                                text_val = item.get("text", "")
                                if not isinstance(text_val, str):
                                    text_val = str(text_val)
                                new_content.append(types.TextContent(type="text", text=text_val))
                            elif item_type == "image":
                                new_content.append(
                                    types.ImageContent(
                                        type="image",
                                        data=item.get("data", ""),
                                        mimeType=item.get("mimeType", "image/png"),
                                    )
                                )
                            elif item_type == "resource":
                                resource = item.get("resource")
                                if not isinstance(resource, dict):
                                    raise TypeError(
                                        f"Expected dict for 'resource', got {type(resource).__name__}"
                                    )
                                new_content.append(
                                    types.EmbeddedResource(type="resource", resource=resource)
                                )
                            else:
                                # Unknown content type: preserve as text to avoid silent data loss
                                logger.warning(
                                    "Unknown MCP content type '%s'; falling back to TextContent",
                                    item_type,
                                )
                                new_content.append(types.TextContent(type="text", text=str(item)))
                        except Exception as item_exc:
                            # Construction of a typed MCP object failed; preserve raw item text
                            logger.warning(
                                "Failed to construct MCP type for item type '%s': %s. "
                                "Falling back to raw TextContent.",
                                item_type,
                                item_exc,
                            )
                            new_content.append(types.TextContent(type="text", text=str(item)))

                    try:
                        original_result.content = new_content
                        return original_result
                    except (AttributeError, TypeError) as assign_exc:
                        logger.warning(
                            "Could not assign reconstructed content to result object: %s. "
                            "Returning intercepted dict result.",
                            assign_exc,
                        )
                        return intercepted_result

                except ImportError:
                    logger.debug("mcp package not available; returning intercepted dict result")
                except Exception as exc:
                    logger.warning(
                        "Unexpected error during MCP result unwrap: %s. "
                        "Returning intercepted dict result.",
                        exc,
                    )
        return intercepted_result

    def enable(self) -> None:
        """Enable interception by replacing session.call_tool"""
        self._session.call_tool = self.call_tool
        logger.info("MCP session interception enabled")

    def disable(self) -> None:
        """Disable interception by restoring original session.call_tool"""
        self._session.call_tool = self._original_call_tool
        logger.info("MCP session interception disabled")


def intercept_mcp_session(session: Any, launchpad: "Launchpad") -> MCPClientInterceptor:
    """
    Create and enable a session-level interceptor for an MCP client

    An alternative to launching individual tool shuttles — intercepts at the
    session.call_tool level so all tools are covered automatically.

    Args:
        session: MCP client session object
        launchpad: Launchpad instance for interception

    Returns:
        MCPClientInterceptor instance (already enabled)

    Example:
        ```python
        from orbit import Launchpad, StationCache
        from orbit.shuttles.mcp_shuttle import intercept_mcp_session

        launchpad = Launchpad(StationCache())
        interceptor = intercept_mcp_session(client.session, launchpad)

        # Now all tool calls through session.call_tool are intercepted
        result = await client.session.call_tool("my_tool", {"arg": "value"})
        ```
    """
    interceptor = MCPClientInterceptor(session, launchpad)
    interceptor.enable()
    return interceptor
