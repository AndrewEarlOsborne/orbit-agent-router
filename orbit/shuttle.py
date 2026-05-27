"""Shuttle: the payload unit that travels between a Launchpad and a Station"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Shuttle:
    """
    A data payload in transit within the orbit system.

    Created by the Launchpad when a tool call result is intercepted. Carries the
    full payload and its metadata from the agent's surface to a Station, where it
    docks and is stored. The LLM receives a manifest instead of the full payload.
    """

    tool_call_id: str
    tool_name: str
    payload: Any
    content: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = self.payload if isinstance(self.payload, (dict, list)) else str(self.payload)
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "payload": payload,
            "content": self.content,
        }
