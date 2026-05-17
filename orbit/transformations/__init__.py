"""
Orbit Transformations Framework

Data transformation layer for AI agents. Enables transformations on stored data
without loading it into LLM context. Agents work with references and summaries only.
"""

from orbit.transformations.base import DataType, TransformationRegistry
from orbit.transformations.decorators import orbit_transformation_tool_mcp
from orbit.transformations.resources import ResourceManager

__version__ = "0.1.0"

__all__ = [
    "DataType",
    "TransformationRegistry",
    "orbit_transformation_tool_mcp",
    "ResourceManager",
]
