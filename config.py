"""
config.py
---------
Central configuration: paths, model name, and environment variables.
Loads a .env file if present (see .env.example).

This project uses Google Gemini only -- via the free tier at
https://aistudio.google.com/apikey (no credit card required). There is no
Anthropic/Claude dependency anywhere in this codebase.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.getenv("DATA_PATH", PROJECT_ROOT / "data" / "retail_sales_india.csv"))
CHARTS_DIR = Path(os.getenv("CHARTS_DIR", PROJECT_ROOT / "charts"))
CHARTS_DIR.mkdir(exist_ok=True, parents=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# gemini-3.6-flash is Google's current workhorse Flash model (shipped 21 Jul
# 2026): free via the AI Studio free tier, tool-calling capable, 1M-token
# context. If a newer Flash model has since replaced it, just change
# MODEL_NAME in .env -- no code changes needed. Check the current lineup at
# https://ai.google.dev/gemini-api/docs/models
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.6-flash")

# Safety / control limits for the agent loop
# MAX_AGENT_ITERATIONS caps how many reasoning/tool-call steps the Query
# Agent gets (CrewAI's Agent.max_iter) -- it needs roughly one iteration per
# planned sub-question, plus a few spare for self-correction after an SQL
# error, so the default is generous rather than tight.
MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "15"))
MAX_ROWS_RETURNED = int(os.getenv("MAX_ROWS_RETURNED", "500"))
SQL_TABLE_NAME = "sales"


def has_api_key() -> bool:
    return bool(GOOGLE_API_KEY)
