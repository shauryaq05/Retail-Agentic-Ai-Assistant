

from dataclasses import dataclass, field
from typing import List, Optional

from crewai.tools import tool

from app import data_tool


@dataclass
class TraceEntry:
    sql: str
    ok: bool
    error: Optional[str] = None
    columns: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False


trace: List[TraceEntry] = []


def reset_trace() -> None:
    trace.clear()


@tool("execute_sql")
def execute_sql_tool(sql: str) -> str:
    """Execute a read-only SQL SELECT query against the `sales` table of
    Indian multi-outlet retail transactions and return the resulting rows as
    text. Only SELECT/WITH statements are permitted -- anything else
    (INSERT/UPDATE/DELETE/DROP/etc.) is rejected before it reaches the
    database."""
    result = data_tool.execute_sql(sql)
    trace.append(
        TraceEntry(
            sql=sql,
            ok=result.ok,
            error=result.error,
            columns=result.columns,
            rows=result.rows,
            row_count=result.row_count,
            truncated=result.truncated,
        )
    )
    return result.to_tool_result_text()
