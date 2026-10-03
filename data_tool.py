"""
data_tool.py
------------
This is the "Tool/function calling to a SQL execution tool" piece of the
architecture (Unit III). It loads the retail sales CSV into an in-memory
SQLite database once, and exposes a single guarded function, `execute_sql`,
that the LLM agent calls as a tool. It also exposes `get_schema_description`
so the planner/query agent can ground its SQL generation in the real schema
instead of guessing column names.

Design choices:
- Read-only guard: only SELECT statements are allowed. Anything else
  (INSERT/UPDATE/DELETE/DROP/ATTACH/PRAGMA, etc.) is rejected before it
  reaches SQLite, so a hallucinated or adversarial query can't mutate data.
- Row cap: results are capped (MAX_ROWS_RETURNED) to keep responses small
  and to stop a runaway query from blowing up the context window.
- Errors are returned as structured strings (not raised) so the agent loop
  can feed the error back to the LLM and let it self-correct.
"""

import re
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from app import config

_FORBIDDEN_PATTERN = re.compile(
    r"\b(insert|update|delete|drop|alter|attach|detach|pragma|create|replace|vacuum)\b",
    re.IGNORECASE,
)

_lock = threading.Lock()
_connection: Optional[sqlite3.Connection] = None
_dataframe: Optional[pd.DataFrame] = None


@dataclass
class SqlResult:
    ok: bool
    sql: str
    rows: list = field(default_factory=list)
    columns: list = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    error: Optional[str] = None

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.columns)

    def to_tool_result_text(self) -> str:
        """Compact text representation to feed back to the LLM."""
        if not self.ok:
            return f"SQL_ERROR: {self.error}"
        preview = self.as_dataframe().head(20).to_string(index=False)
        note = " (truncated preview, first 20 of %d rows)" % self.row_count if self.row_count > 20 else ""
        return f"OK | rows={self.row_count}{note}\n{preview}"


def load_data(csv_path=None) -> None:
    """Loads the CSV into an in-memory SQLite table. Safe to call multiple times."""
    global _connection, _dataframe
    with _lock:
        if _connection is not None:
            return
        path = csv_path or config.DATA_PATH
        df = pd.read_csv(path, parse_dates=["date"])
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        df.to_sql(config.SQL_TABLE_NAME, conn, index=False, if_exists="replace")
        _connection = conn
        _dataframe = df


def get_dataframe() -> pd.DataFrame:
    if _dataframe is None:
        load_data()
    return _dataframe


def get_schema_description() -> str:
    """Human/LLM-readable schema description used to ground SQL generation."""
    df = get_dataframe()
    lines = [f"Table `{config.SQL_TABLE_NAME}` ({len(df):,} rows). Columns:"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_vals = df[col].dropna().unique()[:4]
        sample_str = ", ".join(str(v) for v in sample_vals)
        lines.append(f"  - {col} ({dtype}): e.g. {sample_str}")
    lines.append(
        "\nNotes: 'date' spans "
        f"{df['date'].min().date()} to {df['date'].max().date()}. "
        "'sales_amount' is net revenue after discount. 'profit' and "
        "'profit_margin_pct' are already computed per transaction. "
        "'city_tier' is one of Tier 1 / Tier 2 / Tier 3."
    )
    return "\n".join(lines)


def execute_sql(sql: str) -> SqlResult:
    """
    Executes a read-only SQL SELECT query against the sales table.
    This is the single function exposed to the LLM as a callable tool.
    """
    if _connection is None:
        load_data()

    cleaned = sql.strip().rstrip(";")
    if not re.match(r"^\s*(select|with)\b", cleaned, re.IGNORECASE):
        return SqlResult(ok=False, sql=sql, error="Only SELECT/WITH statements are allowed.")
    if _FORBIDDEN_PATTERN.search(cleaned):
        return SqlResult(ok=False, sql=sql, error="Query contains a forbidden keyword (write/DDL operations are blocked).")

    try:
        cur = _connection.cursor()
        cur.execute(cleaned)
        columns = [d[0] for d in cur.description] if cur.description else []
        all_rows = cur.fetchall()
        truncated = len(all_rows) > config.MAX_ROWS_RETURNED
        rows = all_rows[: config.MAX_ROWS_RETURNED]
        rows_as_dicts = [dict(zip(columns, r)) for r in rows]
        return SqlResult(
            ok=True,
            sql=cleaned,
            rows=rows_as_dicts,
            columns=columns,
            row_count=len(all_rows),
            truncated=truncated,
        )
    except sqlite3.Error as e:
        return SqlResult(ok=False, sql=cleaned, error=str(e))


# --- Gemini tool-use (function-calling) schema for this function -------------
SQL_TOOL_SCHEMA = {
    "name": "execute_sql",
    "description": (
        "Execute a read-only SQL SELECT query against the `sales` table of "
        "Indian multi-outlet retail transactions and return the resulting rows. "
        "Only SELECT/WITH statements are permitted."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "A single SQLite-compatible SELECT statement.",
            }
        },
        "required": ["sql"],
    },
}
