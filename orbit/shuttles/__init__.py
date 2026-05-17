"""Shuttle classes for different agent frameworks"""

from orbit.shuttles.mcp_shuttle import launch_mcp_shuttle
from orbit.shuttles.langchain_shuttle import launch_langchain_shuttle

__all__ = ["launch_mcp_shuttle", "launch_langchain_shuttle"]
