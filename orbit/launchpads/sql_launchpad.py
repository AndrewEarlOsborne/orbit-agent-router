"""SQLLaunchpad: intercepts LangChain SQLDatabaseToolkit results and materializes them to DuckDB.

Requires: pip install orbit[sql]
"""

import ast
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from orbit.launchpad import Launchpad
from orbit.station import Station

logger = logging.getLogger(__name__)

# Tool name emitted by LangChain's QuerySQLDataBaseTool
_SQL_QUERY_TOOL = "sql_db_query"


@dataclass
class SQLResultDescriptor:
    """Metadata stored alongside the materialized DuckDB result file."""

    tool_call_id: str
    tool_name: str
    source_connection: str
    # Best-effort: populated if the launchpad is given query context externally.
    # #TODO: capture query from tool input args by overriding _process_result and
    # plumbing tool input through alongside tool output.
    last_query: str
    materialized_path: str
    column_names: List[str]
    row_count: int


class SQLLaunchpad(Launchpad):
    """
    Launchpad subclass for LangChain SQLDatabaseToolkit tools.

    When sql_db_query fires, intercepts the text result, parses it into rows,
    and materializes it to a local DuckDB file keyed by tool_call_id.  The LLM
    receives a compact summary (schema + row count + preview) instead of the raw
    result string.  Other SQL toolkit tools (schema, list, checker) fall through
    to default threshold-based masking.

    The DuckDB file contains two tables:
      result              — the query rows
      _orbit_descriptor   — metadata (connection string, row count, column names, …)

    Transformation tools in sql_transforms.py operate on the DuckDB file via
    the resource_uri returned in the summary.

    Args:
        station:            Storage backend for raw payloads (default: StationCache).
        output_dir:         Directory where .duckdb result files are written.
        source_connection:  SQLAlchemy connection string of the source DB.
                            Stored in the descriptor so reconnect transforms can re-query.
        threshold:          Character limit for masking non-query tool results.
        preview_rows:       Number of rows to include in the LLM-facing summary.
    """

    def __init__(
        self,
        station: Optional[Station] = None,
        output_dir: str = "./orbit_sql_results",
        source_connection: str = "",
        threshold: int = 2048,
        preview_rows: int = 5,
    ) -> None:
        super().__init__(station)
        self._output_dir = output_dir
        self._source_connection = source_connection
        self._threshold = threshold
        self._preview_rows = preview_rows
        os.makedirs(output_dir, exist_ok=True)
        logger.info(
            "SQLLaunchpad initialized: output_dir=%s, preview_rows=%d",
            output_dir,
            preview_rows,
        )

    # ------------------------------------------------------------------
    # Launchpad abstract implementation
    # ------------------------------------------------------------------

    def _generate_summary(
        self, tool_call_id: str, tool_name: str, content: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if tool_name == _SQL_QUERY_TOOL:
            text = self._extract_text(content)
            if text is not None:
                return self._materialize_and_summarize(tool_call_id, tool_name, text)
        return self._mask_content_default(content)

    # ------------------------------------------------------------------
    # SQL materialization
    # ------------------------------------------------------------------

    def _materialize_and_summarize(
        self, tool_call_id: str, tool_name: str, text: str
    ) -> List[Dict[str, Any]]:
        rows, col_names = self._parse_result_text(text)

        if rows is None:
            logger.warning("Could not parse SQL result text for tool_call_id: %s", tool_call_id)
            preview = text[:200] + ("..." if len(text) > 200 else "")
            return [{"type": "text", "text": f"SQL result (unparseable). Raw preview:\n{preview}"}]

        db_path = os.path.join(self._output_dir, f"{tool_call_id}.duckdb")
        self._write_duckdb(db_path, rows, col_names, tool_call_id, tool_name)

        descriptor = SQLResultDescriptor(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            source_connection=self._source_connection,
            last_query="",
            materialized_path=db_path,
            column_names=col_names,
            row_count=len(rows),
        )
        logger.info(
            "Materialized SQL result: %d rows, %d cols -> %s",
            len(rows),
            len(col_names),
            db_path,
        )
        return self._build_summary(descriptor, rows)

    def _parse_result_text(self, text: str) -> Tuple[Optional[List[List[Any]]], List[str]]:
        """
        Parse LangChain SQLDatabase.run() output into (rows, col_names).

        Handles:
          - List of dicts:  [{"col": val, ...}, ...]  (include_columns=True)
          - List of tuples: [(val, ...), ...]          (default)
          - Single scalars: [val, ...]
          - Empty:          [] / "" / "None"
          Returns (None, []) when parsing fails entirely.
        """
        text = text.strip()
        if not text or text in ("None", "[]", "()", ""):
            return [], []

        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None, []

        if not isinstance(parsed, list):
            return None, []

        if not parsed:
            return [], []

        first = parsed[0]

        if isinstance(first, dict):
            col_names = list(first.keys())
            rows: List[List[Any]] = [[row.get(c) for c in col_names] for row in parsed]
            return rows, col_names

        if isinstance(first, (tuple, list)):
            n_cols = len(first)
            col_names = [f"col_{i}" for i in range(n_cols)]
            rows = [list(item) for item in parsed]
            return rows, col_names

        # Single-column scalar list
        return [[item] for item in parsed], ["result"]

    def _write_duckdb(
        self,
        db_path: str,
        rows: List[List[Any]],
        col_names: List[str],
        tool_call_id: str,
        tool_name: str,
    ) -> None:
        try:
            import duckdb
        except ImportError:
            raise ImportError(
                "duckdb is required for SQLLaunchpad. Install with: pip install orbit[sql]"
            )

        con = duckdb.connect(db_path)
        try:
            if col_names and rows:
                # DuckDB infers types automatically via VALUES — use a temp relation.
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
                ("source_connection", self._source_connection),
                ("last_query", ""),
                ("column_names", json.dumps(col_names)),
                ("row_count", str(len(rows))),
            ]
            con.execute("CREATE TABLE _orbit_descriptor (key VARCHAR PRIMARY KEY, value VARCHAR)")
            con.executemany("INSERT INTO _orbit_descriptor VALUES (?, ?)", descriptor_rows)
        finally:
            con.close()

    def _build_summary(
        self, descriptor: SQLResultDescriptor, rows: List[List[Any]]
    ) -> List[Dict[str, Any]]:
        col_names = descriptor.column_names
        preview = rows[: self._preview_rows]

        lines: List[str] = [
            f"SQL result materialized: {descriptor.row_count} rows, {len(col_names)} columns",
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
    # Default masking fallback (for non-query SQL tools)
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
                return item.get("text", "")
        return None


def _sql_literal(value: Any) -> str:
    """Render a Python value as a DuckDB SQL literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    # Escape single quotes by doubling them
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"
