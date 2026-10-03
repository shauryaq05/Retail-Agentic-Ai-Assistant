

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from crewai import Agent, Crew, LLM, Process, Task

from app import config, data_tool
from app.planner import Plan, build_plan
from app.tools import execute_sql_tool, reset_trace, trace

_llm: Optional[LLM] = None


def _get_llm() -> LLM:
    """Lazily builds the shared CrewAI LLM (Gemini via litellm). Cached
    after first call; kept as its own function so tests can patch it."""
    global _llm
    if _llm is None:
        if not config.has_api_key():
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey "
                "and put it in your .env file."
            )
        # litellm (which CrewAI's LLM class wraps) expects provider-prefixed
        # model ids, e.g. "gemini/gemini-3.6-flash".
        _llm = LLM(model=f"gemini/{config.MODEL_NAME}", api_key=config.GOOGLE_API_KEY, temperature=0.2)
    return _llm


def build_query_agent() -> Agent:
    return Agent(
        role="Retail SQL Analyst",
        goal=(
            "Answer each analytical sub-question by writing and executing exactly the SQL "
            "needed, using the execute_sql tool."
        ),
        backstory=(
            "You are a meticulous SQL analyst at an Indian multi-outlet retail chain. You "
            "never guess at numbers -- you always run a query and read the actual result "
            "before stating a figure. If a query errors, you read the error and fix the SQL."
        ),
        tools=[execute_sql_tool],
        llm=_get_llm(),
        max_iter=config.MAX_AGENT_ITERATIONS,
        verbose=False,
        allow_delegation=False,
    )


def build_report_agent() -> Agent:
    return Agent(
        role="BI Reporting Analyst",
        goal=(
            "Turn verified query results into a clear, accurate answer for a non-technical "
            "business stakeholder."
        ),
        backstory=(
            "You write the final answer a retail stakeholder actually reads. You only use "
            "numbers that were actually returned by a query, and you clearly flag anything "
            "that could not be fully verified instead of stating it as fact."
        ),
        llm=_get_llm(),
        verbose=False,
        allow_delegation=False,
    )


@dataclass
class VerificationReport:
    index: int
    passed: bool
    issues: List[str] = field(default_factory=list)


def run_query_stage(plan: Plan, question: str) -> str:
    """Runs the Query Agent over the full plan as one Task; it calls the SQL
    tool once per sub-question (or more, if it needs to self-correct after
    an error). Returns the agent's own narrative summary of what it found."""
    reset_trace()
    schema_text = data_tool.get_schema_description()
    steps_text = "\n".join(f"{s.step_id}. {s.description}" for s in plan.steps)

    agent = build_query_agent()
    task = Task(
        description=(
            f"Schema:\n{schema_text}\n\n"
            f"Overall business question: {question}\n\n"
            "Answer EACH of the following sub-questions with exactly one execute_sql call per "
            f"sub-question (more only if a query errors and you need to fix it):\n{steps_text}\n\n"
            "After every query succeeds, note the actual numeric result -- never invent a number "
            "that wasn't in a query's returned rows."
        ),
        expected_output=(
            "A numbered list, one entry per sub-question above, each stating the sub-question and "
            "the actual numeric result returned by its query."
        ),
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    return result.raw


def verify_trace() -> List[VerificationReport]:
    """Deterministic (no LLM call) grounding checks over every SQL call the
    Query Agent actually made. This is intentionally code, not another model
    call: for a result that has to be trustworthy, code checking code is
    more reliable than a model checking itself."""
    reports: List[VerificationReport] = []
    for i, entry in enumerate(trace):
        issues: List[str] = []
        if not entry.ok:
            issues.append(f"Query failed: {entry.error}")
        else:
            if entry.row_count == 0:
                issues.append("Query returned zero rows.")
            if entry.row_count > 0:
                df = pd.DataFrame(entry.rows, columns=entry.columns)
                numeric_cols = df.select_dtypes(include="number").columns
                for col in numeric_cols:
                    if df[col].isna().all():
                        issues.append(f"Column '{col}' is entirely NULL in the result.")
            if entry.row_count > config.MAX_ROWS_RETURNED * 0.9:
                issues.append(
                    "Result set is unusually large -- the query may be missing an "
                    "aggregation (GROUP BY)."
                )
        reports.append(VerificationReport(index=i, passed=len(issues) == 0, issues=issues))
    return reports


def run_report_stage(question: str, narrative: str, verifications: List[VerificationReport]) -> str:
    caveats = [f"Query {v.index + 1}: {'; '.join(v.issues)}" for v in verifications if not v.passed]
    caveats_text = (
        "\n".join(f"- {c}" for c in caveats)
        if caveats
        else "None -- every query was verified against its actual returned data."
    )

    agent = build_report_agent()
    task = Task(
        description=(
            f"Original business question: {question}\n\n"
            f"Query Agent's findings:\n{narrative}\n\n"
            f"Verification caveats (queries that could NOT be fully verified):\n{caveats_text}\n\n"
            "Write the final answer: 3-6 sentences, plain business language, no SQL or column "
            "names. Lead with the direct answer. If the question was comparative, say explicitly "
            "which side is higher/lower and by how much. If a caveat applies to a number you use, "
            "say so explicitly instead of stating it as fact."
        ),
        expected_output="A short, plain-English answer for a business stakeholder.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    return result.raw


def _build_chart() -> Optional[str]:
    """Best-effort: picks the first captured query result that looks
    chartable (a categorical/time column plus a numeric column) and plots
    it. Operates on the real captured trace, not on anything the LLM wrote."""
    for entry in trace:
        if not (entry.ok and entry.row_count >= 2):
            continue
        df = pd.DataFrame(entry.rows, columns=entry.columns)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
        if not numeric_cols or not non_numeric_cols:
            continue

        label_col, value_col = non_numeric_cols[0], numeric_cols[0]
        if df[label_col].nunique() > 30:
            continue  # too many categories to be a readable chart

        fig, ax = plt.subplots(figsize=(7, 4.5))
        is_time = any(k in label_col.lower() for k in ("date", "month", "year"))
        if is_time:
            ax.plot(df[label_col].astype(str), df[value_col], marker="o")
            plt.xticks(rotation=45, ha="right")
        else:
            ax.bar(df[label_col].astype(str), df[value_col], color="#4C72B0")
            plt.xticks(rotation=30, ha="right")
        ax.set_xlabel(label_col)
        ax.set_ylabel(value_col)
        ax.set_title(f"{value_col} by {label_col}")
        fig.tight_layout()

        filename = f"chart_{uuid.uuid4().hex[:8]}.png"
        out_path = config.CHARTS_DIR / filename
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return str(out_path)
    return None


def ask(question: str) -> dict:
    """Top-level entry point: runs the full pipeline and returns a dict
    shaped exactly like the API's AskResponse, so main.py just does
    `AskResponse(**ask(question))`."""
    plan = build_plan(question)
    narrative = run_query_stage(plan, question)
    verifications = verify_trace()
    chart_path = _build_chart()
    answer = run_report_stage(question, narrative, verifications)

    v_by_index = {v.index: v for v in verifications}
    steps = []
    for i, entry in enumerate(trace):
        v = v_by_index.get(i)
       
        description = plan.steps[i].description if i < len(plan.steps) else f"Additional query {i + 1}"
        steps.append(
            {
                "step_id": i + 1,
                "description": description,
                "sql": entry.sql,
                "verified": bool(v and v.passed),
                "issues": (v.issues if v else []),
                "row_count": entry.row_count,
                "columns": entry.columns,
                "rows": entry.rows,
                "observation": "",
            }
        )

    caveats = [f"Query {v.index + 1}: {'; '.join(v.issues)}" for v in verifications if not v.passed]
    chart_url = f"/charts/{Path(chart_path).name}" if chart_path else None

    return {
        "question": question,
        "is_comparative": plan.is_comparative,
        "steps": steps,
        "answer": answer,
        "caveats": caveats,
        "chart_url": chart_url,
    }
