"""DuckDBLaunchpad: materializes structured tool payloads to DuckDB.

Requires: pip install orbit[sql]

Usage with fastmcp (minimal pattern):
    from orbit.launchpads.duckdb_launchpad import DuckDBLaunchpad

    launchpad = DuckDBLaunchpad(output_dir="./results")

    @server.tool()
    async def my_tool(query: str) -> list[dict]:
        ...

    shuttle = launchpad.stage(my_tool)
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

from orbit.launchpad import Launchpad
from orbit.station import Station

logger = logging.getLogger(__name__)


@dataclass
class DuckDBResultDescriptor:
    """Metadata stored alongside a materialized DuckDB result file."""

    tool_call_id: str
    tool_name: str
    materialized_path: str
    column_names: List[str]
    row_count: int


class DuckDBLaunchpad(Launchpad):
    """
    Launchpad that materializes structured tool results to local DuckDB files.

    Any tool launched as a shuttle via stage() has its output intercepted: if the result
    content contains parseable tabular data (JSON array of objects, arrays, or
    scalars), it is written to a .duckdb file keyed by tool_call_id.  The LLM
    receives a compact summary (schema + row count + preview) instead of the
    full payload.  Non-tabular content falls through to threshold-based masking.

    The DuckDB file contains two tables:
      result             — the materialized rows
      _orbit_descriptor  — metadata (tool_call_id, tool_name, row count, column names)

    Transformation tools in sql_transforms.py operate on the .duckdb file via
    the resource_uri returned in the summary.

    Args:
        station:      Storage backend for raw payloads (default: StationCache).
        output_dir:   Directory where .duckdb result files are written.
        threshold:    Character limit for masking non-tabular content.
        preview_rows: Number of rows to include in the LLM-facing summary.
    """

    def __init__(
        self,
        station: Optional[Station] = None,
        output_dir: str = "./orbit_duckdb_results",
        threshold: int = 2048,
        preview_rows: int = 5,
    ) -> None:
        super().__init__(station)
        self._output_dir = output_dir
        self._threshold = threshold
        self._preview_rows = preview_rows
        os.makedirs(output_dir, exist_ok=True)
        logger.info("DuckDBLaunchpad initialized: output_dir=%s", output_dir)

    # ------------------------------------------------------------------
    # Launchpad abstract implementation
    # ------------------------------------------------------------------

    def _generate_summary(
        self, tool_call_id: str, tool_name: str, content: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        text = self._extract_text(content)
        if text is not None:
            rows, col_names = self._parse_tabular(text)
            if rows is not None:
                return self._materialize_and_summarize(tool_call_id, tool_name, rows, col_names)
        return self._mask_content_default(content)

    # ------------------------------------------------------------------
    # Tabular parsing
    # ------------------------------------------------------------------

    def _parse_tabular(self, text: str) -> Tuple[Optional[List[List[Any]]], List[str]]:
        """
        Attempt to parse text as tabular data.

        Handles:
          - JSON array of dicts:   [{"col": val, ...}, ...]
          - JSON array of arrays:  [[val, ...], ...]
          - JSON array of scalars: [val, ...]
          - Python literal forms via ast.literal_eval as fallback

        Returns (None, []) when text is not parseable as tabular data.
        """
        text = text.strip()
        if not text:
            return None, []

        parsed: Any = None
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass

        if parsed is None:
            import ast

            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return None, []

        if not isinstance(parsed, list) or not parsed:
            return None, []

        first = parsed[0]

        if isinstance(first, dict):
            col_names = list(first.keys())
            rows: List[List[Any]] = [[row.get(c) for c in col_names] for row in parsed]
            return rows, col_names

        if isinstance(first, (list, tuple)):
            col_names = [f"col_{i}" for i in range(len(first))]
            rows = [list(item) for item in parsed]
            return rows, col_names

        return [[item] for item in parsed], ["result"]

    # ------------------------------------------------------------------
    # DuckDB materialization
    # ------------------------------------------------------------------

    def _materialize_and_summarize(
        self,
        tool_call_id: str,
        tool_name: str,
        rows: List[List[Any]],
        col_names: List[str],
    ) -> List[Dict[str, Any]]:
        db_path = os.path.join(self._output_dir, f"{tool_call_id}.duckdb")
        self._write_duckdb(db_path, rows, col_names, tool_call_id, tool_name)

        descriptor = DuckDBResultDescriptor(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            materialized_path=db_path,
            column_names=col_names,
            row_count=len(rows),
        )
        logger.info(
            "Materialized to DuckDB: %d rows, %d cols -> %s",
            len(rows),
            len(col_names),
            db_path,
        )
        return self._build_summary(descriptor, rows)

    def _write_duckdb(
        self,
        db_path: str,
        rows: List[List[Any]],
        col_names: List[str],
        tool_call_id: str,
        tool_name: str,
    ) -> None:
        duckdb = _require_duckdb()
        con = duckdb.connect(db_path)
        try:
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

            descriptor_rows = [
                ("tool_call_id", tool_call_id),
                ("tool_name", tool_name),
                ("column_names", json.dumps(col_names)),
                ("row_count", str(len(rows))),
            ]
            con.execute("CREATE TABLE _orbit_descriptor (key VARCHAR PRIMARY KEY, value VARCHAR)")
            con.executemany("INSERT INTO _orbit_descriptor VALUES (?, ?)", descriptor_rows)
        finally:
            con.close()

    def _build_summary(
        self, descriptor: DuckDBResultDescriptor, rows: List[List[Any]]
    ) -> List[Dict[str, Any]]:
        col_names = descriptor.column_names
        preview = rows[: self._preview_rows]

        lines: List[str] = [
            f"Result materialized to DuckDB: {descriptor.row_count} rows, {len(col_names)} columns",
            f"Resource URI: {descriptor.materialized_path}",
            f"Columns: {col_names}",
            "",
            f"Preview (first {len(preview)} rows):",
        ]

        if col_names:
            col_widths = [
                max(len(str(c)), max((len(str(row[i])) for row in preview), default=0))
                for i, c in enumerate(col_names)
            ]
            header = " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(col_names))
            separator = "-+-".join("-" * w for w in col_widths)
            lines += [header, separator]
            for row in preview:
                lines.append(
                    " | ".join(
                        str(v if v is not None else "NULL").ljust(col_widths[i])
                        for i, v in enumerate(row)
                    )
                )

        return [{"type": "text", "text": "\n".join(lines)}]

    # ------------------------------------------------------------------
    # Default masking fallback (non-tabular content)
    # ------------------------------------------------------------------

    def _mask_content_default(self, content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        masked: List[Dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                masked.append(item)
                continue
            masked_item: Dict[str, Any] = {}
            for k, v in item.items():
                if isinstance(v, str) and len(v) > self._threshold:
                    masked_item[k] = {
                        "original_type": "string",
                        "length": len(v),
                        "summary": f"Content masked - exceeds {self._threshold} chars",
                        "preview": v[:100] + "...",
                    }
                else:
                    masked_item[k] = v
            masked.append(masked_item)
        return masked

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_text(self, content: List[Dict[str, Any]]) -> Optional[str]:
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return cast(Optional[str], item.get("text", ""))
        return None


# ------------------------------------------------------------------
# Module-level helpers (shared with sql_transforms.py)
# ------------------------------------------------------------------


def _require_duckdb() -> Any:
    try:
        import duckdb

        return duckdb
    except ImportError:
        raise ImportError(
            "duckdb is required for DuckDBLaunchpad. Install with: pip install orbit[sql]"
        )


def _sql_literal(value: Any) -> str:
    """Render a Python value as a DuckDB SQL literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"
