Agentic BI  Retail Assistant(Team - 1)

A small project that lets you ask plain-English questions about retail sales data
("Which region had the highest sales last year?") and get back an answer with the
SQL that was run, a verification check, and a chart.

There's a backend (FastAPI) that does all the work, and a simple frontend (plain HTML/CSS/JS)
that talks to it.

 How it works

1. You ask a question.
2. A planner (LangChain + Gemini) breaks it down into smaller steps if needed.
3. A CrewAI agent writes SQL for each step and runs it against a SQLite database.
4. The results are checked (empty results, missing columns, etc.) before being trusted.
5. A second agent writes the final answer in plain English, and a chart is generated if it makes sense.

Uses Google Gemini (`gemini-3.6-flash`) since it's free. No Anthropic/Claude used anywhere.

 Project structure

```
backend/     FastAPI app + the agent logic (LangChain + CrewAI)
frontend/    static site that calls the backend
```


## Data

The dataset (`backend/data/retail_sales_india.csv`) is synthetic data generated to look like
real Indian retail sales data — region, city tier, category, discounts, profit, etc. You can
regenerate it with more rows using `backend/data/generate_dataset.py`.

## Notes

- Only SELECT queries are allowed, so the AI can't modify the database.
- CORS is wide open in the backend for local dev — tighten it before deploying anywhere.
