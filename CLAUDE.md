# PulseIQ — Project Context

Near real-time economic stress platform.
FastAPI backend · Next.js frontend · DuckDB · Airflow

Stack: Python 3.11, dbt, XGBoost, LangChain,
ChromaDB, Ollama (lfm2.5-thinking), deck.gl

Load detail files with @ when needed:
  @CLAUDE_PIPELINE.md  → DAG + dbt rules
  @CLAUDE_API.md       → FastAPI + contracts
  @CLAUDE_FRONTEND.md  → React + design tokens
  @CLAUDE_RAG.md       → RAG + Ollama setup
  @CLAUDE_ML.md        → Model + SHAP + calibration

Never touch: data/, models/, airflow/logs/
Never use print() — always logging module
Never hardcode secrets — always os.getenv()
Show diff before applying any change.