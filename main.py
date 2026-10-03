

from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import config, data_tool
from app.crew import ask as run_pipeline


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Loads the CSV into the in-memory SQLite table once, at process start,
    # instead of on the first request.
    data_tool.load_data()
    yield


app = FastAPI(
    title="Agentic BI Assistant API",
    description="Ask natural-language questions about retail sales data and get a verified, plain-English answer.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/charts", StaticFiles(directory=str(config.CHARTS_DIR)), name="charts")



class AskRequest(BaseModel):
    question: str


class StepTrace(BaseModel):
    step_id: int
    description: str
    sql: Optional[str] = None
    verified: bool
    issues: List[str] = []
    row_count: Optional[int] = None
    columns: Optional[List[str]] = None
    rows: Optional[list] = None
    observation: str = ""


class AskResponse(BaseModel):
    question: str
    is_comparative: bool
    steps: List[StepTrace]
    answer: str
    caveats: List[str]
    chart_url: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
    api_key_configured: bool


# --- Endpoints ----------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        provider="gemini",
        model=config.MODEL_NAME,
        api_key_configured=config.has_api_key(),
    )


@app.get("/schema")
def schema() -> dict:
    return {"schema": data_tool.get_schema_description()}


@app.get("/examples")
def examples() -> dict:
    return {
        "questions": [
            "Which product category had declining profit margins in North India last year?",
            "Compare average discount rates between Tier 1 and Tier 2 cities",
            "What are the total sales by region in 2024?",
            "Which sub-category has the highest profit margin in Electronics?",
            "How did monthly sales in the Electronics category trend over 2024?",
        ]
    }


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="`question` must not be empty.")

    if not config.has_api_key():
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY is not configured on the server (.env).",
        )

    try:
        result = run_pipeline(question)
    except Exception as e:  # noqa: BLE001 -- surfaced to the API caller as a 500
        raise HTTPException(status_code=500, detail=f"Agent pipeline failed: {e}") from e

    # app.crew.ask() already returns a dict shaped exactly like AskResponse.
    return AskResponse(**result)
