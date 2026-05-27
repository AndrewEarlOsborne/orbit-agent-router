"""Example SQL transformation functions for Orbit, backed by DuckDB.

These tools operate on .duckdb files produced by DuckDBLaunchpad.  The pattern
mirrors csv_transforms.py: each function receives resource_uri (the source
.duckdb file), applies a transform, and writes to either the same file
(in_place=True) or a new sibling file.

Requires: pip install orbit[sql]
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from orbit.transformations.base import DataType
from orbit.transformations.decorators import orbit_transformation_tool_mcp

logger = logging.getLogger(__name__)


def _sql_output_path(resource_uri: str, transform_name: str, in_place: bool) -> str:
    """Compute destination path: same file for in-place, new sibling .duckdb otherwise."""
    if in_place:
        return resource_uri
    base, _ = os.path.splitext(resource_uri)
    return f"{base}.{transform_name}.duckdb"


def _require_duckdb() -> Any:
    try:
        import duckdb

        return duckdb
    except ImportError:
        raise ImportError(
            "duckdb is required for SQL transforms. Install with: pip install orbit[sql]"
        )


def _copy_descriptor(src_path: str, dst_path: str, overrides: Dict[str, str]) -> None:
    """Attach both DuckDB files and copy _orbit_descriptor, applying key overrides."""
    duckdb = _require_duckdb()
    con = duckdb.connect()
    con.execute(f"ATTACH '{src_path}' AS src (READ_ONLY)")
    con.execute(f"ATTACH '{dst_path}' AS dst")
    con.execute("CREATE TABLE dst._orbit_descriptor AS SELECT * FROM src._orbit_descriptor")
    for key, value in overrides.items():
        con.execute(
            "UPDATE dst._orbit_descriptor SET value = ? WHERE key = ?",
            [value, key],
        )
    con.close()


def _read_descriptor(resource_uri: str) -> Dict[str, str]:
    duckdb = _require_duckdb()
    con = duckdb.connect(resource_uri, read_only=True)
    try:
        rows = con.execute("SELECT key, value FROM _orbit_descriptor").fetchall()
        return {k: v for k, v in rows}
    finally:
        con.close()


# ------------------------------------------------------------------
# Config models
# ------------------------------------------------------------------


class SQLFilterConfig(BaseModel):
    where_clause: str


class SQLAggregateConfig(BaseModel):
    group_cols: List[str]
    agg_col: str
    agg_func: str  # sum | avg | count | min | max


class SQLExportConfig(BaseModel):
    format: str = "csv"  # csv | parquet
    output_path: Optional[str] = None


class SQLReconnectConfig(BaseModel):
    query: str  # SQL query to execute against source DB


# ------------------------------------------------------------------
# Transformation tools
# ------------------------------------------------------------------


@orbit_transformation_tool_mcp(
    data_type=DataType.SQL,
    description="Filter a materialized SQL result by a SQL WHERE clause",
    transform_config=SQLFilterConfig,
)
async def filter_sql(
    resource_uri: str,
    where_clause: str,
    in_place: bool = False,
) -> Dict[str, Any]:
    """
    Filter the materialized result table using a WHERE clause.

    Args:
        resource_uri: Path to .duckdb file produced by DuckDBLaunchpad (source).
        where_clause: SQL WHERE expression, e.g. "revenue > 1000 AND region = 'US'".
        in_place:     If True, replace result table in-place.
                      If False, write a new sibling .duckdb file.

    Returns:
        Summary with filtered row count and output resource URI.
    """
    try:
        duckdb = _require_duckdb()
        out_path = _sql_output_path(resource_uri, "filter_sql", in_place)

        total = (
            duckdb.connect(resource_uri, read_only=True)
            .execute("SELECT COUNT(*) FROM result")
            .fetchone()[0]
        )
        if in_place:
            con = duckdb.connect(resource_uri)
            try:
                con.execute(f"CREATE TEMP TABLE _tmp AS SELECT * FROM result WHERE {where_clause}")
                con.execute("DELETE FROM result")
                con.execute("INSERT INTO result SELECT * FROM _tmp")
                filtered_count = con.execute("SELECT COUNT(*) FROM result").fetchone()[0]
            finally:
                con.close()
        else:
            con = duckdb.connect()
            try:
                con.execute(f"ATTACH '{resource_uri}' AS src (READ_ONLY)")
                con.execute(f"ATTACH '{out_path}' AS dst")
                con.execute(
                    f"CREATE TABLE dst.result AS SELECT * FROM src.result WHERE {where_clause}"
                )
                filtered_count = con.execute("SELECT COUNT(*) FROM dst.result").fetchone()[0]
            finally:
                con.close()
            _copy_descriptor(resource_uri, out_path, {"row_count": str(filtered_count)})

        logger.info("filter_sql: %d -> %d rows (WHERE %s)", total, filtered_count, where_clause)
        return {
            "type": "text",
            "text": (
                f"Filtered result: {filtered_count} of {total} rows match WHERE {where_clause}. "
                f"Output: {out_path}"
            ),
        }
    except Exception as e:
        logger.error("filter_sql error: %s", e)
        return {"type": "text", "text": f"Error in filter_sql: {e}"}


@orbit_transformation_tool_mcp(
    data_type=DataType.SQL,
    description="Aggregate a materialized SQL result with GROUP BY",
    transform_config=SQLAggregateConfig,
)
async def aggregate_sql(
    resource_uri: str,
    group_cols: List[str],
    agg_col: str,
    agg_func: str = "sum",
    in_place: bool = False,
) -> Dict[str, Any]:
    """
    Apply a GROUP BY aggregation to the materialized result table.

    Args:
        resource_uri: Path to .duckdb file produced by DuckDBLaunchpad (source).
        group_cols:   Columns to group by.
        agg_col:      Column to aggregate.
        agg_func:     Aggregation function: sum | avg | count | min | max.
        in_place:     If True, replace result table in-place.
                      If False, write a new sibling .duckdb file.

    Returns:
        Summary with group count and output resource URI.
    """
    _VALID_FUNCS = {"sum", "avg", "count", "min", "max"}
    if agg_func not in _VALID_FUNCS:
        return {
            "type": "text",
            "text": f"Invalid agg_func '{agg_func}'. Must be one of {_VALID_FUNCS}",
        }

    try:
        duckdb = _require_duckdb()

        group_expr = ", ".join(f'"{c}"' for c in group_cols)
        agg_alias = f"{agg_func}_{agg_col}"
        select_expr = f'{group_expr}, {agg_func}("{agg_col}") AS "{agg_alias}"'
        out_path = _sql_output_path(resource_uri, "aggregate_sql", in_place)

        if in_place:
            con = duckdb.connect(resource_uri)
            try:
                con.execute(
                    f"CREATE TEMP TABLE _tmp AS "
                    f"SELECT {select_expr} FROM result GROUP BY {group_expr}"
                )
                con.execute("DROP TABLE result")
                con.execute("CREATE TABLE result AS SELECT * FROM _tmp")
                group_count = con.execute("SELECT COUNT(*) FROM result").fetchone()[0]
            finally:
                con.close()
        else:
            con = duckdb.connect()
            try:
                con.execute(f"ATTACH '{resource_uri}' AS src (READ_ONLY)")
                con.execute(f"ATTACH '{out_path}' AS dst")
                con.execute(
                    f"CREATE TABLE dst.result AS "
                    f"SELECT {select_expr} FROM src.result GROUP BY {group_expr}"
                )
                group_count = con.execute("SELECT COUNT(*) FROM dst.result").fetchone()[0]
            finally:
                con.close()
            new_cols = json.dumps(group_cols + [agg_alias])
            _copy_descriptor(
                resource_uri,
                out_path,
                {"row_count": str(group_count), "column_names": new_cols},
            )

        logger.info(
            "aggregate_sql: %d groups, %s(%s) by %s", group_count, agg_func, agg_col, group_cols
        )
        return {
            "type": "text",
            "text": (
                f"Aggregated result: {group_count} groups. "
                f"{agg_func}({agg_col}) grouped by {group_cols}. "
                f"Output: {out_path}"
            ),
        }
    except Exception as e:
        logger.error("aggregate_sql error: %s", e)
        return {"type": "text", "text": f"Error in aggregate_sql: {e}"}


@orbit_transformation_tool_mcp(
    data_type=DataType.SQL,
    description="Export a materialized SQL result to CSV or Parquet",
    transform_config=SQLExportConfig,
)
async def export_sql(
    resource_uri: str,
    format: str = "csv",
    output_path: Optional[str] = None,
    in_place: bool = False,
) -> Dict[str, Any]:
    """
    Export the materialized result table to CSV or Parquet.

    Args:
        resource_uri: Path to .duckdb file produced by DuckDBLaunchpad (source).
        format:       Output format: "csv" or "parquet".
        output_path:  Destination file path. Defaults to resource_uri with new extension.
        in_place:     Ignored — export always writes a new file.

    Returns:
        Summary with output file path and row count.
    """
    _VALID_FORMATS = {"csv", "parquet"}
    if format not in _VALID_FORMATS:
        return {"type": "text", "text": f"Invalid format '{format}'. Must be csv or parquet."}

    try:
        duckdb = _require_duckdb()

        if output_path is None:
            base, _ = os.path.splitext(resource_uri)
            output_path = f"{base}.{format}"

        duckdb_format = "CSV" if format == "csv" else "PARQUET"
        header_opt = ", HEADER" if format == "csv" else ""

        con = duckdb.connect(resource_uri, read_only=True)
        try:
            row_count = con.execute("SELECT COUNT(*) FROM result").fetchone()[0]
            con.execute(f"COPY result TO '{output_path}' (FORMAT {duckdb_format}{header_opt})")
        finally:
            con.close()

        logger.info("export_sql: %d rows -> %s (%s)", row_count, output_path, format)
        return {
            "type": "text",
            "text": f"Exported {row_count} rows to {format.upper()}: {output_path}",
        }
    except Exception as e:
        logger.error("export_sql error: %s", e)
        return {"type": "text", "text": f"Error in export_sql: {e}"}


@orbit_transformation_tool_mcp(
    data_type=DataType.SQL,
    description=(
        "Re-execute a SQL query against the original source database and materialize the new result"
    ),
    transform_config=SQLReconnectConfig,
)
async def reconnect_and_query(
    resource_uri: str,
    query: str,
    in_place: bool = False,
) -> Dict[str, Any]:
    """
    Re-run (or run a new) SQL query against the source database recorded in the descriptor.

    Reads source_connection from _orbit_descriptor, executes the given query via
    SQLAlchemy, and materializes the result to a new (or the same) .duckdb file.

    Args:
        resource_uri: Path to .duckdb file whose descriptor contains source_connection.
        query:        SQL query to run against the source database.
        in_place:     If True, replace result table in-place.
                      If False, write a new sibling .duckdb file.

    Returns:
        Summary with new row count and output resource URI.

    Requires:
        pip install orbit[sql] <db-driver>
    """
    try:
        import sqlalchemy as sa  # type: ignore[import-not-found]
    except ImportError:
        return {
            "type": "text",
            "text": "reconnect_and_query requires sqlalchemy. Install with: pip install orbit[sql]",
        }

    try:
        duckdb = _require_duckdb()
        descriptor = _read_descriptor(resource_uri)
        source_connection = descriptor.get("source_connection", "")

        if not source_connection:
            return {
                "type": "text",
                "text": (
                    "No source_connection in descriptor. "
                    # TODO: store source_connection in _orbit_descriptor at materialization time
                    # to enable reconnect transforms.
                    "source_connection must be present in the DuckDB descriptor to reconnect."
                ),
            }

        engine = sa.create_engine(source_connection)
        with engine.connect() as conn:
            proxy = conn.execute(sa.text(query))
            col_names: List[str] = list(proxy.keys())
            rows: List[List[Any]] = [list(r) for r in proxy.fetchall()]

        out_path = _sql_output_path(resource_uri, "reconnect_and_query", in_place)
        _write_rows_to_duckdb(duckdb, out_path, rows, col_names, in_place)

        overrides = {
            "last_query": query,
            "row_count": str(len(rows)),
            "column_names": json.dumps(col_names),
        }
        if not in_place:
            _copy_descriptor(resource_uri, out_path, overrides)
        else:
            con = duckdb.connect(resource_uri)
            try:
                for key, value in overrides.items():
                    con.execute(
                        "UPDATE _orbit_descriptor SET value = ? WHERE key = ?",
                        [value, key],
                    )
            finally:
                con.close()

        logger.info("reconnect_and_query: %d rows from %s", len(rows), source_connection)
        return {
            "type": "text",
            "text": (
                f"Re-queried source database: {len(rows)} rows, columns {col_names}. "
                f"Output: {out_path}"
            ),
        }
    except Exception as e:
        logger.error("reconnect_and_query error: %s", e)
        return {"type": "text", "text": f"Error in reconnect_and_query: {e}"}


# ------------------------------------------------------------------
# Internal helper shared with DuckDBLaunchpad
# ------------------------------------------------------------------


def _write_rows_to_duckdb(
    duckdb: Any,
    db_path: str,
    rows: List[List[Any]],
    col_names: List[str],
    in_place: bool,
) -> None:
    from orbit.launchpads.duckdb_launchpad import _sql_literal

    con = duckdb.connect(db_path)
    try:
        if in_place:
            con.execute("DROP TABLE IF EXISTS result")

        if col_names and rows:
            sanitized = [f'"{c}"' for c in col_names]
            col_list = ", ".join(sanitized)
            value_rows = ["(" + ", ".join(_sql_literal(v) for v in row) + ")" for row in rows]
            values_clause = ", ".join(value_rows)
            con.execute(
                f"CREATE TABLE result AS SELECT * FROM (VALUES {values_clause}) t({col_list})"
            )
        else:
            con.execute("CREATE TABLE result (_empty VARCHAR)")
    finally:
        con.close()
