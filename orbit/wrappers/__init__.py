"""Tool wrappers for different frameworks"""

from orbit.wrappers.mcp_wrapper import wrap_mcp_tool
from orbit.wrappers.langchain_wrapper import wrap_langchain_tool

__all__ = ["wrap_mcp_tool", "wrap_langchain_tool"]
