"""Example CSV transformation functions for Orbit in the MCP Protocol"""

import csv
import os
import logging
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel

from orbit.transformations.decorators import orbit_transformation_tool_mcp
from orbit.transformations.base import DataType

logger = logging.getLogger(__name__)


def _csv_output_path(resource_uri: str, transform_name: str, in_place: bool) -> str:
    """Compute destination path: same file for in-place, new sibling file otherwise."""
    if in_place:
        return resource_uri
    base, ext = os.path.splitext(resource_uri)
    return f"{base}.{transform_name}{ext}"


# ------------------------------------------------------------------
# Config models
# ------------------------------------------------------------------


class FilterConfig(BaseModel):
    filter_col: str
    operator: str  # "==", ">", "<", ">=", "<=", "!=", "in", "not_in"
    value: Any
    case_sensitive: bool = False


class SelectConfig(BaseModel):
    columns: List[str]


class RenameConfig(BaseModel):
    rename_map: Dict[str, str]  # old_name -> new_name


class GroupByConfig(BaseModel):
    group_cols: List[str]
    agg_col: str
    agg_func: str  # "sum", "avg", "count", "min", "max"


# ------------------------------------------------------------------
# Transformation tools
# ------------------------------------------------------------------


@orbit_transformation_tool_mcp(
    data_type=DataType.CSV,
    description="Filter CSV rows by a column condition",
    transform_config=FilterConfig,
)
async def filter_csv(
    resource_uri: str,
    filter_col: str,
    operator: str,
    value: Any,
    case_sensitive: bool = False,
    in_place: bool = False,
) -> Dict[str, Any]:
    """
    Filter CSV rows by column condition.

    Args:
        resource_uri:   Path to CSV file (source).
        filter_col:     Column name to filter on.
        operator:       Comparison operator: ==, >, <, >=, <=, !=, in, not_in.
        value:          Value to compare against.
        case_sensitive: Whether string comparison is case-sensitive.
        in_place:       If True, overwrite source. If False, write sibling file.

    Returns:
        Summary with filtered row count and output path.
    """
    try:
        rows: List[Dict[str, Any]] = []
        fieldnames: List[str] = []

        with open(resource_uri, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            rows = list(reader)

        if filter_col not in fieldnames:
            return {
                "type": "text",
                "text": f"Error: Column '{filter_col}' not found. Available: {fieldnames}",
            }

        filtered: List[Dict[str, Any]] = []
        for row in rows:
            col_value: Any = row.get(filter_col, "")
            if not case_sensitive and isinstance(col_value, str):
                col_value = col_value.lower()
                compare_value = str(value).lower() if isinstance(value, str) else value
            else:
                compare_value = value

            try:
                match = False
                if operator == "==":
                    match = col_value == compare_value
                elif operator == ">":
                    match = float(col_value) > float(compare_value)
                elif operator == "<":
                    match = float(col_value) < float(compare_value)
                elif operator == ">=":
                    match = float(col_value) >= float(compare_value)
                elif operator == "<=":
                    match = float(col_value) <= float(compare_value)
                elif operator == "!=":
                    match = col_value != compare_value
                elif operator == "in":
                    match = isinstance(value, (list, tuple)) and col_value in value
                elif operator == "not_in":
                    match = isinstance(value, (list, tuple)) and col_value not in value
                if match:
                    filtered.append(row)
            except (ValueError, TypeError):
                continue

        output_path = _csv_output_path(resource_uri, "filter_csv", in_place)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            if filtered and fieldnames:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(filtered)

        logger.info("filter_csv: %d -> %d rows, output: %s", len(rows), len(filtered), output_path)
        return {
            "type": "text",
            "text": (
                f"Filtered {len(filtered)} of {len(rows)} rows "
                f"where {filter_col} {operator} {value}. "
                f"Columns: {fieldnames}. Output: {output_path}"
            ),
        }

    except Exception as e:
        logger.error("filter_csv error: %s", e)
        return {"type": "text", "text": f"Error: {e}"}


@orbit_transformation_tool_mcp(
    data_type=DataType.CSV,
    description="Select specific columns from a CSV file",
    transform_config=SelectConfig,
)
async def select_csv(
    resource_uri: str,
    columns: List[str],
    in_place: bool = False,
) -> Dict[str, Any]:
    """
    Select specific columns from CSV.

    Args:
        resource_uri: Path to CSV file (source).
        columns:      Column names to keep.
        in_place:     If True, overwrite source. If False, write sibling file.

    Returns:
        Summary with selected columns and row count.
    """
    try:
        rows: List[Dict[str, Any]] = []
        fieldnames: List[str] = []

        with open(resource_uri, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            rows = list(reader)

        missing = set(columns) - set(fieldnames)
        if missing:
            return {
                "type": "text",
                "text": f"Error: Columns not found: {missing}. Available: {fieldnames}",
            }

        selected = [{col: row.get(col, "") for col in columns} for row in rows]

        output_path = _csv_output_path(resource_uri, "select_csv", in_place)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(selected)

        logger.info(
            "select_csv: %d rows, %d cols, output: %s", len(selected), len(columns), output_path
        )
        return {
            "type": "text",
            "text": (
                f"Selected {len(selected)} rows with {len(columns)} columns: {columns}. "
                f"Output: {output_path}"
            ),
        }

    except Exception as e:
        logger.error("select_csv error: %s", e)
        return {"type": "text", "text": f"Error: {e}"}


@orbit_transformation_tool_mcp(
    data_type=DataType.CSV,
    description="Rename columns in a CSV file",
    transform_config=RenameConfig,
)
async def rename_csv(
    resource_uri: str,
    rename_map: Dict[str, str],
    in_place: bool = False,
) -> Dict[str, Any]:
    """
    Rename columns in CSV.

    Args:
        resource_uri: Path to CSV file (source).
        rename_map:   Mapping of old column name -> new column name.
        in_place:     If True, overwrite source. If False, write sibling file.

    Returns:
        Summary with renamed columns and new schema.
    """
    try:
        rows: List[Dict[str, Any]] = []
        fieldnames: List[str] = []

        with open(resource_uri, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            rows = list(reader)

        missing = set(rename_map.keys()) - set(fieldnames)
        if missing:
            return {"type": "text", "text": f"Error: Columns to rename not found: {missing}"}

        new_fieldnames = [rename_map.get(col, col) for col in fieldnames]
        renamed_rows = [
            {new_fieldnames[i]: row[old] for i, old in enumerate(fieldnames)} for row in rows
        ]

        output_path = _csv_output_path(resource_uri, "rename_csv", in_place)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=new_fieldnames)
            writer.writeheader()
            writer.writerows(renamed_rows)

        logger.info("rename_csv: renamed %d cols, output: %s", len(rename_map), output_path)
        return {
            "type": "text",
            "text": (
                f"Renamed {len(rename_map)} columns: {rename_map}. "
                f"New schema: {new_fieldnames}. Output: {output_path}"
            ),
        }

    except Exception as e:
        logger.error("rename_csv error: %s", e)
        return {"type": "text", "text": f"Error: {e}"}


@orbit_transformation_tool_mcp(
    data_type=DataType.CSV,
    description="Group CSV by columns and compute aggregate",
    transform_config=GroupByConfig,
)
async def group_by_csv(
    resource_uri: str,
    group_cols: List[str],
    agg_col: str,
    agg_func: str = "sum",
    in_place: bool = False,
) -> Dict[str, Any]:
    """
    Group CSV by columns and compute an aggregate.

    Args:
        resource_uri: Path to CSV file (source).
        group_cols:   Columns to group by.
        agg_col:      Column to aggregate.
        agg_func:     Aggregation function: sum, avg, count, min, max.
        in_place:     If True, overwrite source. If False, write sibling file.

    Returns:
        Summary with group count and output path.
    """
    try:
        from collections import defaultdict

        rows: List[Dict[str, Any]] = []
        fieldnames: List[str] = []

        with open(resource_uri, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            rows = list(reader)

        missing = set(group_cols + [agg_col]) - set(fieldnames)
        if missing:
            return {"type": "text", "text": f"Error: Columns not found: {missing}"}

        agg_map = {
            "sum": sum,
            "avg": lambda x: sum(x) / len(x) if x else 0,
            "count": len,
            "min": lambda x: min(x) if x else 0,
            "max": lambda x: max(x) if x else 0,
        }
        if agg_func not in agg_map:
            return {"type": "text", "text": f"Error: Unknown agg_func: {agg_func}"}

        groups: Dict[Tuple[Any, ...], List[float]] = defaultdict(list)
        for row in rows:
            key = tuple(row.get(col, "") for col in group_cols)
            try:
                groups[key].append(float(row.get(agg_col, 0)))
            except (ValueError, TypeError):
                continue

        result_fieldnames = group_cols + [f"{agg_func}_{agg_col}"]
        result_rows = [
            {**dict(zip(group_cols, key)), f"{agg_func}_{agg_col}": agg_map[agg_func](vals)}  # type: ignore[no-untyped-call]
            for key, vals in groups.items()
        ]

        output_path = _csv_output_path(resource_uri, "group_by_csv", in_place)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=result_fieldnames)
            writer.writeheader()
            writer.writerows(result_rows)

        logger.info(
            "group_by_csv: %d groups, %s(%s), output: %s",
            len(result_rows),
            agg_func,
            agg_col,
            output_path,
        )
        return {
            "type": "text",
            "text": (
                f"Grouped by {group_cols}, computed {agg_func}({agg_col}): "
                f"{len(result_rows)} groups. Schema: {result_fieldnames}. Output: {output_path}"
            ),
        }

    except Exception as e:
        logger.error("group_by_csv error: %s", e)
        return {"type": "text", "text": f"Error: {e}"}


@orbit_transformation_tool_mcp(
    data_type=DataType.CSV,
    description="Count rows in a CSV file",
)
async def count_csv(
    resource_uri: str,
    in_place: bool = False,
) -> Dict[str, Any]:
    """
    Count rows in CSV. Does not modify data; in_place is ignored.

    Args:
        resource_uri: Path to CSV file.
        in_place:     Ignored — count never modifies data.

    Returns:
        Summary with row count and schema.
    """
    try:
        count = 0
        fieldnames: List[str] = []

        with open(resource_uri, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            count = sum(1 for _ in reader)

        logger.info("count_csv: %d rows, %d columns", count, len(fieldnames))
        return {
            "type": "text",
            "text": f"CSV has {count} rows with {len(fieldnames)} columns: {fieldnames}",
        }

    except Exception as e:
        logger.error("count_csv error: %s", e)
        return {"type": "text", "text": f"Error: {e}"}
