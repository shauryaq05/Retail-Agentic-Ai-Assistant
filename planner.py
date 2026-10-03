

from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app import config, data_tool



class PlanStep(BaseModel):
    step_id: int = Field(description="1-based step number, in execution order")
    description: str = Field(description="A single, concrete aggregation or lookup, in plain English")


class Plan(BaseModel):
    is_comparative: bool = Field(description="True if the question compares two or more groups")
    final_goal: str = Field(description="One sentence describing what the final answer must contain")
    steps: List[PlanStep]


_structured_llm = None


def _get_structured_llm():
    """Lazily builds the LangChain chat model + structured-output wrapper.
    Cached after first call. Kept as its own function (rather than inlined
    into build_plan) so tests can patch it with a fake that returns a canned
    Plan, without needing a real API key or network access."""
    global _structured_llm
    if _structured_llm is None:
        if not config.has_api_key():
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. "
                "and put it in your .env file."
            )
        llm = ChatGoogleGenerativeAI(
            model=config.MODEL_NAME,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0.1,
        )
        _structured_llm = llm.with_structured_output(Plan)
    return _structured_llm


def build_plan(question: str) -> Plan:
    schema_text = data_tool.get_schema_description()
    structured_llm = _get_structured_llm()
    plan: Plan = structured_llm.invoke(
        [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"Schema:\n{schema_text}\n\nBusiness question: {question}"),
        ]
    )
    return plan
