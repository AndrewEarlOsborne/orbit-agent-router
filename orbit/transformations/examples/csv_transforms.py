"""Example CSV transformation functions for Orbit in the MCP Protocol"""

import csv
import logging
from typing import Any, Dict, List
from pydantic import BaseModel

from orbit.transformations.decorators import orbit_transformation_tool_mcp
from orbit.transformations.base import DataType

logger = logging.getLogger(__name__)


# Pydantic config models for type-safe parameters
class FilterConfig(BaseModel):
    """Configuration for CSV filtering"""

    filter_col: str
    operator: str  # "==", ">", "<", ">=", "<=", "!=", "in", "not_in"
    value: Any
    case_sensitive: bool = False


class SelectConfig(BaseModel):
    """Configuration for column selection"""

    columns: List[str]


class RenameConfig(BaseModel):
    """Configuration for column renaming"""

    rename_map: Dict[str, str]  # old_name -> new_name


class GroupByConfig(BaseModel):
    """Configuration for grouping and aggregation"""

    group_cols: List[str]
    agg_col: str
    agg_func: str  # "sum", "avg", "count", "min", "max"


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
    Filter CSV rows by column condition

    Args:
        resource_uri: Path to CSV file
        filter_col: Column name to filter on
        operator: Comparison operator (==, >, <, >=, <=, !=, in, not_in)
        value: Value to compare against
        case_sensitive: Whether string comparison is case-sensitive
        in_place: Whether to update original file or create new one

    Returns:
        Summary with filtered row count and schema
    """
    try:
        rows = []
        fieldnames = []

        # Read CSV
        with open(resource_uri, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames = reader.fieldnames
            rows = [row for row in reader]

        # Validate filter column exists
        if filter_col not in fieldnames:
            return {
                "type": "text",
                "text": f"Error: Column '{filter_col}' not found in CSV. Available: {fieldnames}",
            }

        # Apply filter
        filtered = []
        for row in rows:
            col_value = row.get(filter_col, "")

            # Prepare values for comparison
            if not case_sensitive and isinstance(col_value, str):
                col_value = col_value.lower()
                compare_value = str(value).lower() if isinstance(value, str) else value
            else:
                compare_value = value

            # Apply operator
            try:
                if operator == "==":
                    if col_value == compare_value:
                        filtered.append(row)
                elif operator == ">":
                    if float(col_value) > float(compare_value):
                        filtered.append(row)
                elif operator == "<":
                    if float(col_value) < float(compare_value):
                        filtered.append(row)
                elif operator == ">=":
                    if float(col_value) >= float(compare_value):
                        filtered.append(row)
                elif operator == "<=":
                    if float(col_value) <= float(compare_value):
                        filtered.append(row)
                elif operator == "!=":
                    if col_value != compare_value:
                        filtered.append(row)
                elif operator == "in":
                    if isinstance(value, (list, tuple)):
                        if col_value in value:
                            filtered.append(row)
                elif operator == "not_in":
                    if isinstance(value, (list, tuple)):
                        if col_value not in value:
                            filtered.append(row)
            except (ValueError, TypeError):
                # Skip rows that can't be compared
                continue

        # Write filtered CSV
        with open(resource_uri, "w", newline="", encoding="utf-8") as f:
            if filtered and fieldnames:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(filtered)

        logger.info("Filtered CSV: %d rows -> %d rows", len(rows), len(filtered))

        return {
            "type": "text",
            "text": f"Filtered {len(filtered)} rows (from {len(rows)}) where {filter_col} {operator} {value}. Columns: {fieldnames}",
        }

    except Exception as e:
        logger.error("Error filtering CSV: %s", e)
        return {"type": "text", "text": f"Error: {str(e)}"}


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
    Select specific columns from CSV

    Args:
        resource_uri: Path to CSV file
        columns: List of column names to keep
        in_place: Whether to update original file or create new one

    Returns:
        Summary with selected columns and row count
    """
    try:
        rows = []
        fieldnames = []

        # Read CSV
        with open(resource_uri, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames = reader.fieldnames
            rows = [row for row in reader]

        # Validate columns exist
        missing = set(columns) - set(fieldnames)
        if missing:
            return {
                "type": "text",
                "text": f"Error: Columns not found: {missing}. Available: {fieldnames}",
            }

        # Select columns
        selected = []
        for row in rows:
            selected_row = {col: row.get(col, "") for col in columns}
            selected.append(selected_row)

        # Write selected CSV
        with open(resource_uri, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(selected)

        logger.info("Selected %d columns from CSV", len(columns))

        return {
            "type": "text",
            "text": f"Selected {len(selected)} rows with {len(columns)} columns: {columns}",
        }

    except Exception as e:
        logger.error("Error selecting columns from CSV: %s", e)
        return {"type": "text", "text": f"Error: {str(e)}"}


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
    Rename columns in CSV

    Args:
        resource_uri: Path to CSV file
        rename_map: Dictionary mapping old column names to new names
        in_place: Whether to update original file or create new one

    Returns:
        Summary with renamed columns
    """
    try:
        rows = []
        fieldnames = []

        # Read CSV
        with open(resource_uri, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            rows = [row for row in reader]

        # Validate columns to rename exist
        missing = set(rename_map.keys()) - set(fieldnames)
        if missing:
            return {
                "type": "text",
                "text": f"Error: Columns to rename not found: {missing}",
            }

        # Rename columns in fieldnames
        new_fieldnames = [rename_map.get(col, col) for col in fieldnames]

        # Rename columns in rows
        renamed_rows = []
        for row in rows:
            renamed_row = {new_fieldnames[i]: row[old_col]
                          for i, old_col in enumerate(fieldnames)}
            renamed_rows.append(renamed_row)

        # Write renamed CSV
        with open(resource_uri, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=new_fieldnames)
            writer.writeheader()
            writer.writerows(renamed_rows)

        logger.info("Renamed %d columns in CSV", len(rename_map))

        return {
            "type": "text",
            "text": f"Renamed {len(rename_map)} columns: {rename_map}. New schema: {new_fieldnames}",
        }

    except Exception as e:
        logger.error("Error renaming columns in CSV: %s", e)
        return {"type": "text", "text": f"Error: {str(e)}"}


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
    Group CSV by columns and compute aggregate

    Args:
        resource_uri: Path to CSV file
        group_cols: Columns to group by
        agg_col: Column to aggregate
        agg_func: Aggregation function (sum, avg, count, min, max)
        in_place: Whether to update original file or create new one

    Returns:
        Summary with aggregation results
    """
    try:
        rows = []
        fieldnames = []

        # Read CSV
        with open(resource_uri, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames = reader.fieldnames
            rows = [row for row in reader]

        # Validate columns exist
        missing = set(group_cols + [agg_col]) - set(fieldnames)
        if missing:
            return {
                "type": "text",
                "text": f"Error: Columns not found: {missing}",
            }

        # Group and aggregate
        from collections import defaultdict

        groups = defaultdict(list)
        for row in rows:
            group_key = tuple(row.get(col, "") for col in group_cols)
            try:
                val = float(row.get(agg_col, 0))
                groups[group_key].append(val)
            except (ValueError, TypeError):
                continue

        # Compute aggregates
        agg_map = {
            "sum": sum,
            "avg": lambda x: sum(x) / len(x) if x else 0,
            "count": len,
            "min": lambda x: min(x) if x else 0,
            "max": lambda x: max(x) if x else 0,
        }

        if agg_func not in agg_map:
            return {
                "type": "text",
                "text": f"Error: Unknown aggregation function: {agg_func}",
            }

        agg_func_obj = agg_map[agg_func]

        # Build result rows
        result_fieldnames = group_cols + [f"{agg_func}_{agg_col}"]
        result_rows = []
        for group_key, values in groups.items():
            row = {group_cols[i]: group_key[i] for i in range(len(group_cols))}
            row[f"{agg_func}_{agg_col}"] = agg_func_obj(values)
            result_rows.append(row)

        # Write aggregated CSV
        with open(resource_uri, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=result_fieldnames)
            writer.writeheader()
            writer.writerows(result_rows)

        logger.info(
            "Aggregated CSV: grouped by %s, computed %s(%s), result: %d groups",
            group_cols,
            agg_func,
            agg_col,
            len(result_rows),
        )

        return {
            "type": "text",
            "text": f"Grouped by {group_cols}, computed {agg_func}({agg_col}): {len(result_rows)} groups. Schema: {result_fieldnames}",
        }

    except Exception as e:
        logger.error("Error aggregating CSV: %s", e)
        return {"type": "text", "text": f"Error: {str(e)}"}


@orbit_transformation_tool_mcp(
    data_type=DataType.CSV,
    description="Count rows in a CSV file",
)
async def count_csv(
    resource_uri: str,
    in_place: bool = False,
) -> Dict[str, Any]:
    """
    Count rows in CSV

    Args:
        resource_uri: Path to CSV file
        in_place: Ignored (count doesn't modify data)

    Returns:
        Summary with row count and schema
    """
    try:
        count = 0
        fieldnames = []

        # Count rows
        with open(resource_uri, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames = reader.fieldnames
            count = sum(1 for _ in reader)

        logger.info("Counted CSV: %d rows, %d columns", count, len(fieldnames))

        return {
            "type": "text",
            "text": f"CSV has {count} rows with {len(fieldnames)} columns: {fieldnames}",
        }

    except Exception as e:
        logger.error("Error counting CSV: %s", e)
        return {"type": "text", "text": f"Error: {str(e)}"}
