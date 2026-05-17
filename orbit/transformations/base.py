"""Base classes and types for Orbit Transformations framework"""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DataType(Enum):
    """Supported data types for transformations"""

    CSV = "csv"
    JSON = "json"
    SQL = "sql"
    PARQUET = "parquet"
    TEXT = "text"
    CUSTOM = "custom"


class TransformationMetadata:
    """Metadata for a registered transformation"""

    def __init__(
        self,
        name: str,
        data_type: DataType,
        description: str,
        function: Callable,
        parameters: Dict[str, Any],
    ) -> None:
        """
        Initialize transformation metadata

        Args:
            name: Name of the transformation function
            data_type: Target data type (CSV, JSON, SQL, etc.)
            description: Human-readable description
            function: The actual transformation function
            parameters: Function parameter schema from Pydantic
        """
        self.name = name
        self.data_type = data_type
        self.description = description
        self.function = function
        self.parameters = parameters
        logger.debug("Created TransformationMetadata for %s (%s)", name, data_type.value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation"""
        return {
            "name": self.name,
            "data_type": self.data_type.value,
            "description": self.description,
            "parameters": self.parameters,
        }


class TransformationRegistry:
    """Registry of available transformations by data type"""

    def __init__(self) -> None:
        """Initialize empty transformation registry"""
        self._transforms: Dict[DataType, Dict[str, TransformationMetadata]] = {
            data_type: {} for data_type in DataType
        }
        logger.info("TransformationRegistry initialized")

    def register(
        self,
        name: str,
        data_type: DataType,
        description: str,
        function: Callable,
        parameters: Dict[str, Any],
    ) -> None:
        """
        Register a transformation

        Args:
            name: Name of the transformation
            data_type: Target data type
            description: Human-readable description
            function: The transformation function
            parameters: Function parameters schema

        Raises:
            ValueError: If transformation already registered for this name/type combo
        """
        if name in self._transforms[data_type]:
            raise ValueError(f"Transformation '{name}' already registered for {data_type.value}")

        metadata = TransformationMetadata(
            name=name,
            data_type=data_type,
            description=description,
            function=function,
            parameters=parameters,
        )
        self._transforms[data_type][name] = metadata
        logger.info(
            "Registered transformation '%s' for data type '%s'",
            name,
            data_type.value,
        )

    def get(self, data_type: DataType, name: str) -> Optional[TransformationMetadata]:
        """
        Get a registered transformation by type and name

        Args:
            data_type: Target data type
            name: Transformation name

        Returns:
            TransformationMetadata if found, None otherwise
        """
        return self._transforms.get(data_type, {}).get(name)

    def list_by_type(self, data_type: DataType) -> List[TransformationMetadata]:
        """
        List all transformations for a data type

        Args:
            data_type: Target data type

        Returns:
            List of TransformationMetadata objects
        """
        return list(self._transforms.get(data_type, {}).values())

    def list_all(self) -> Dict[str, List[TransformationMetadata]]:
        """
        List all registered transformations organized by data type

        Returns:
            Dictionary mapping data type names to lists of TransformationMetadata
        """
        return {
            data_type.value: list(transforms.values())
            for data_type, transforms in self._transforms.items()
            if transforms
        }


# Global singleton registry
_global_registry = TransformationRegistry()


def get_registry() -> TransformationRegistry:
    """Get the global transformation registry"""
    return _global_registry
