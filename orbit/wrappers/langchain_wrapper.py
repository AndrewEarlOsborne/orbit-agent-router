"""LangChain tool wrapping functionality"""

from typing import TYPE_CHECKING, Any, Dict, Union
import logging
import uuid

if TYPE_CHECKING:
    from orbit.launchpad import Launchpad

logger = logging.getLogger(__name__)


class LangChainToolWrapper:
    """Wrapper for LangChain tools that intercepts results"""

    def __init__(self, original_tool: Any, launchpad: "Launchpad") -> None:
        """
        Initialize LangChain tool wrapper

        Args:
            original_tool: Original LangChain tool object
            launchpad: Launchpad instance for interception
        """
        self._original_tool = original_tool
        self._launchpad = launchpad

        self.name = original_tool.name
        self.description = original_tool.description
        self.args_schema = getattr(original_tool, "args_schema", None)
        self.return_direct = getattr(original_tool, "return_direct", False)

        logger.debug("Created LangChainToolWrapper for tool: %s", self.name)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """
        Async execution with interception

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Intercepted and summarized result
        """
        tool_call_id = str(uuid.uuid4())
        logger.debug(
            "Executing LangChain tool '%s' (async) with tool_call_id: %s",
            self.name,
            tool_call_id,
        )

        if hasattr(self._original_tool, "_arun"):
            if "config" not in kwargs:
                kwargs["config"] = {}
            result = await self._original_tool._arun(*args, **kwargs)
        elif hasattr(self._original_tool, "ainvoke"):
            input_dict = kwargs if kwargs else (args[0] if args else {})
            result = await self._original_tool.ainvoke(input_dict)
        else:
            raise AttributeError(
                f"Tool {self.name} has no async execution method (_arun or ainvoke)"
            )

        result_wrapped = self._wrap_langchain_result(result)

        intercepted_result = await self._launchpad._process_result(
            tool_call_id, self.name, result_wrapped
        )

        return self._unwrap_result(intercepted_result)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Sync execution with interception

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Intercepted and summarized result
        """
        tool_call_id = str(uuid.uuid4())
        logger.debug(
            "Executing LangChain tool '%s' (sync) with tool_call_id: %s",
            self.name,
            tool_call_id,
        )

        if hasattr(self._original_tool, "_run"):
            if "config" not in kwargs:
                kwargs["config"] = {}
            result = self._original_tool._run(*args, **kwargs)
        elif hasattr(self._original_tool, "invoke"):
            input_dict = kwargs if kwargs else (args[0] if args else {})
            result = self._original_tool.invoke(input_dict)
        else:
            raise AttributeError(
                f"Tool {self.name} has no sync execution method (_run or invoke)"
            )

        result_wrapped = self._wrap_langchain_result(result)

        import asyncio

        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
            intercepted_result = loop.run_until_complete(
                self._launchpad._process_result(tool_call_id, self.name, result_wrapped)
            )
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                intercepted_result = loop.run_until_complete(
                    self._launchpad._process_result(tool_call_id, self.name, result_wrapped)
                )
            finally:
                loop.close()

        return self._unwrap_result(intercepted_result)

    def _wrap_langchain_result(self, result: Any) -> Dict[str, Any]:
        """
        Wrap LangChain result into standard format

        LangChain tools typically return strings or simple objects,
        so we wrap them in a content structure.

        Args:
            result: Raw result from LangChain tool

        Returns:
            Wrapped result in standard format
        """
        if isinstance(result, dict) and "content" in result:
            return result

        if isinstance(result, str):
            return {"content": [{"type": "text", "text": result}]}

        if isinstance(result, (int, float, bool)):
            return {"content": [{"type": "text", "text": str(result)}]}

        if isinstance(result, dict):
            return {"content": [{"type": "data", "data": result}]}

        if isinstance(result, list):
            return {"content": [{"type": "data", "data": {"items": result}}]}

        return {"content": [{"type": "text", "text": str(result)}]}

    def _unwrap_result(self, wrapped_result: Any) -> Any:
        """
        Unwrap result back to LangChain format

        Args:
            wrapped_result: Wrapped result from interception

        Returns:
            Unwrapped result suitable for LangChain
        """
        if isinstance(wrapped_result, dict) and "content" in wrapped_result:
            content = wrapped_result["content"]
            if len(content) == 1:
                item = content[0]
                if item.get("type") == "text":
                    text_val = item.get("text", "")
                    if isinstance(text_val, dict):
                        return str(text_val)
                    return text_val
                elif item.get("type") == "data":
                    data = item.get("data", {})
                    if "items" in data:
                        return data["items"]
                    return data

            combined_text = []
            for item in content:
                if item.get("type") == "text":
                    text_val = item.get("text", "")
                    if isinstance(text_val, dict):
                        combined_text.append(str(text_val))
                    else:
                        combined_text.append(text_val)
                elif item.get("type") == "data":
                    combined_text.append(str(item.get("data", "")))
            return "\n".join(combined_text)

        return wrapped_result

    async def ainvoke(self, input: Union[str, Dict[str, Any]]) -> Any:
        """LangChain-style async invoke method"""
        if isinstance(input, dict):
            return await self._arun(**input)
        return await self._arun(input)

    def invoke(self, input: Union[str, Dict[str, Any]]) -> Any:
        """LangChain-style sync invoke method"""
        if isinstance(input, dict):
            return self._run(**input)
        return self._run(input)


def wrap_langchain_tool(tool: Any, launchpad: "Launchpad") -> Any:
    """
    Wrap a LangChain tool to intercept results

    Args:
        tool: LangChain tool to wrap
        launchpad: Launchpad instance for interception

    Returns:
        Wrapped LangChain tool

    Example:
        ```python
        from langchain_core.tools import tool
        from orbit import Launchpad, StationCache

        @tool
        def my_tool(query: str) -> str:
            '''Search for information'''
            return "result"

        launchpad = Launchpad(StationCache())
        wrapped_tool = wrap_langchain_tool(my_tool, launchpad)
        ```
    """
    return LangChainToolWrapper(tool, launchpad)


class LangChainToolNodeInterceptor:
    """
    Interceptor for LangGraph ToolNode

    This allows interception at the ToolNode level rather than
    wrapping individual tools.
    """

    def __init__(self, tool_node: Any, launchpad: "Launchpad") -> None:
        """
        Initialize ToolNode interceptor

        Args:
            tool_node: LangGraph ToolNode instance
            launchpad: Launchpad instance for interception
        """
        self._tool_node = tool_node
        self._launchpad = launchpad
        self._original_tools = tool_node.tools if hasattr(tool_node, "tools") else []
        logger.info("LangChainToolNodeInterceptor initialized")

    def wrap_tools(self) -> None:
        """Wrap all tools in the ToolNode"""
        if hasattr(self._tool_node, "tools"):
            wrapped_tools = [
                wrap_langchain_tool(tool, self._launchpad) for tool in self._tool_node.tools
            ]
            self._tool_node.tools = wrapped_tools
            logger.info("Wrapped %d tools in ToolNode", len(wrapped_tools))


def intercept_tool_node(tool_node: Any, launchpad: "Launchpad") -> None:
    """
    Intercept a LangGraph ToolNode by wrapping its tools

    Args:
        tool_node: LangGraph ToolNode instance
        launchpad: Launchpad instance for interception

    Example:
        ```python
        from langgraph.prebuilt import ToolNode
        from orbit import Launchpad, StationCache
        from orbit.wrappers.langchain import intercept_tool_node

        tools = [tool1, tool2, tool3]
        tool_node = ToolNode(tools)

        launchpad = Launchpad(StationCache())
        intercept_tool_node(tool_node, launchpad)

        # Now all tools in tool_node are intercepted
        ```
    """
    interceptor = LangChainToolNodeInterceptor(tool_node, launchpad)
    interceptor.wrap_tools()
