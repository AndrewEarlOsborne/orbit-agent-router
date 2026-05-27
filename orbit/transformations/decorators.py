"""Decorators for defining transformations in the Orbit framework"""

import inspect
import functools
from typing import Any, Callable, Dict, Optional, Type, cast
from pydantic import BaseModel, create_model
import logging

from orbit.transformations.base import DataType, get_registry
from orbit.transformations.resources import TransformContext

logger = logging.getLogger(__name__)


def orbit_transformation_tool_mcp(
    data_type: DataType,
    description: str,
    transform_config: Optional[Type[BaseModel]] = None,
) -> Callable[..., Any]:
    """
    Decorator for defining a transformation tool in Orbit

    The decorated function should:
    1. Accept resource_uri as the first parameter (location of data to transform)
    2. Accept in_place as a parameter (controls Station mutation)
    3. Accept transformation-specific parameters
    4. Return a summary dict with {"type": "text", "text": "...summary..."}

    The decorator handles:
    - Extracting tool_call_id from MCP parameters
    - Retrieving resource from Station
    - Passing resource location to your function
    - Handling in-place vs. copy semantics
    - Storing result in Station
    - Generating fastmcp tool definition

    Args:
        data_type: Target data type (DataType.CSV, .JSON, etc.)
        description: Human-readable description for LLM
        transform_config: Optional Pydantic model for validation.
                         If not provided, auto-generated from function signature.

    Returns:
        Decorator function

    Example:
        ```python
        @orbit_transformation_tool_mcp(
            data_type=DataType.CSV,
            description="Filter CSV rows by column condition"
        )
        async def filter_csv(
            resource_uri: str,
            filter_col: str,
            operator: str,
            value: Any,
            in_place: bool = False
        ) -> dict:
            # Load, filter, save, return summary
            ...
        ```
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Extract function signature
        sig = inspect.signature(func)
        func_name = func.__name__
        is_async = inspect.iscoroutinefunction(func)

        # Auto-generate Pydantic config if not provided
        if transform_config is None:
            config = _generate_config_from_signature(sig)
        else:
            config = transform_config

        # Register the transformation
        params_schema = _extract_schema(config) if config else {}
        registry = get_registry()
        registry.register(
            name=func_name,
            data_type=data_type,
            description=description,
            function=func,
            parameters=params_schema,
        )

        logger.info(
            "Registered transformation '%s' for data type '%s'",
            func_name,
            data_type.value,
        )

        # Create wrapper that handles Station integration
        wrapper: Any
        if is_async:

            @functools.wraps(func)
            async def async_wrapper(
                resource_uri: str,
                in_place: bool = False,
                tool_call_id: Optional[str] = None,
                **kwargs: Any,
            ) -> Dict[str, Any]:
                """
                Async wrapper that handles Station integration

                Args:
                    resource_uri: Location of the resource
                    in_place: Whether to mutate original or create new entry
                    tool_call_id: Station reference ID (for integration with Launchpad)
                    **kwargs: Transformation parameters

                Returns:
                    Summary result
                """
                # Validate config if provided
                if config:
                    try:
                        validated = config(**kwargs)
                        kwargs = validated.model_dump()
                    except Exception as e:
                        logger.error(
                            "Failed to validate transform config for '%s': %s",
                            func_name,
                            e,
                        )
                        raise

                # Create transformation context
                ctx = TransformContext(
                    tool_call_id=tool_call_id or "unknown",
                    original_uri=resource_uri,
                    transform_name=func_name,
                    in_place=in_place,
                )

                logger.debug(
                    "Executing transformation '%s' (async) with execution_id: %s",
                    func_name,
                    ctx.execution_id,
                )

                # Pass resource_uri unchanged — functions own their output path.
                # Each transform computes: out = uri if in_place else f"{base}.{name}.ext"
                result = await func(
                    resource_uri=resource_uri,
                    in_place=in_place,
                    **kwargs,
                )

                # Add metadata to result
                if isinstance(result, dict) and "type" in result:
                    result["execution_id"] = ctx.execution_id
                    result["new_tool_call_id"] = ctx.new_tool_call_id
                    result["in_place"] = in_place

                logger.debug(
                    "Completed transformation '%s' (execution_id: %s)",
                    func_name,
                    ctx.execution_id,
                )

                return cast(Dict[str, Any], result)

            wrapper = async_wrapper

        else:

            @functools.wraps(func)
            def sync_wrapper(
                resource_uri: str,
                in_place: bool = False,
                tool_call_id: Optional[str] = None,
                **kwargs: Any,
            ) -> Dict[str, Any]:
                """
                Sync wrapper that handles Station integration

                Args:
                    resource_uri: Location of the resource
                    in_place: Whether to mutate original or create new entry
                    tool_call_id: Station reference ID (for integration with Launchpad)
                    **kwargs: Transformation parameters

                Returns:
                    Summary result
                """
                # Validate config if provided
                if config:
                    try:
                        validated = config(**kwargs)
                        kwargs = validated.model_dump()
                    except Exception as e:
                        logger.error(
                            "Failed to validate transform config for '%s': %s",
                            func_name,
                            e,
                        )
                        raise

                # Create transformation context
                ctx = TransformContext(
                    tool_call_id=tool_call_id or "unknown",
                    original_uri=resource_uri,
                    transform_name=func_name,
                    in_place=in_place,
                )

                logger.debug(
                    "Executing transformation '%s' (sync) with execution_id: %s",
                    func_name,
                    ctx.execution_id,
                )

                # Pass resource_uri unchanged — functions own their output path.
                result = func(
                    resource_uri=resource_uri,
                    in_place=in_place,
                    **kwargs,
                )

                # Add metadata to result
                if isinstance(result, dict) and "type" in result:
                    result["execution_id"] = ctx.execution_id
                    result["new_tool_call_id"] = ctx.new_tool_call_id
                    result["in_place"] = in_place

                logger.debug(
                    "Completed transformation '%s' (execution_id: %s)",
                    func_name,
                    ctx.execution_id,
                )

                return cast(Dict[str, Any], result)

            wrapper = sync_wrapper

        # Attach metadata to wrapper for fastmcp introspection
        wrapper._orbit_transform = True
        wrapper._orbit_data_type = data_type
        wrapper._orbit_description = description
        wrapper._orbit_config = config
        wrapper._orbit_parameters = params_schema

        return cast(Callable[..., Any], wrapper)

    return decorator


def _generate_config_from_signature(sig: inspect.Signature) -> Optional[Type[BaseModel]]:
    """
    Auto-generate Pydantic model from function signature

    Extracts parameters (excluding resource_uri, in_place, tool_call_id)
    and creates a Pydantic model for validation.

    Args:
        sig: Function signature

    Returns:
        Pydantic model class or None if no parameters
    """
    excluded = {"resource_uri", "in_place", "tool_call_id", "self"}
    fields = {}

    for param_name, param in sig.parameters.items():
        if param_name in excluded:
            continue

        # Determine type
        if param.annotation == inspect.Parameter.empty:
            param_type = Any
        else:
            param_type = param.annotation

        # Determine default
        if param.default == inspect.Parameter.empty:
            fields[param_name] = (param_type, ...)
        else:
            fields[param_name] = (param_type, param.default)

    if not fields:
        return None

    return create_model("GeneratedTransformConfig", **cast(Any, fields))


def _extract_schema(config: Type[BaseModel]) -> Dict[str, Any]:
    """
    Extract JSON schema from Pydantic model

    Args:
        config: Pydantic model class

    Returns:
        Dictionary representation of schema
    """
    if config is None:
        return {}

    try:
        schema = config.model_json_schema()
        return {
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }
    except Exception as e:
        logger.warning("Failed to extract schema from config: %s", e)
        return {}
