"""Shuttle: the payload unit that travels between a Launchpad and a Station"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Shuttle:
    """
    A data payload in transit within the orbit system.

    A Shuttle is created by the Launchpad when a tool call result is intercepted.
    It carries the full payload and its metadata from the agent's surface to a
    Station, where it docks and is stored. The LLM receives a manifest instead
    of the full payload.

    Attributes:
        tool_call_id: Unique identifier for the tool call that produced this shuttle
        tool_name:    Name of the tool that was called
        payload:      Full original result from the tool call
        content:      Extracted content list from the payload (list[dict])
    """

    tool_call_id: str
    tool_name: str
    payload: Any
    content: List[Dict[str, Any]] = field(default_factory=list)
