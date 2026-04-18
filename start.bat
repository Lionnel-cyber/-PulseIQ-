@echo off
echo Starting PulseIQ (full stack)...
echo.

:: Infrastructure (Airflow, MLflow, Postgres)
echo [1/3] Starting infrastructure via Docker Compose...
cd /d %~dp0
docker compose up -d
echo.

:: FastAPI backend
echo [2/3] Starting API...
start "PulseIQ API" cmd /k "cd /d %~dp0 && uvicorn src.api.main:app --port 8000"

:: Next.js frontend
echo [3/3] Starting frontend...
start "PulseIQ Frontend" cmd /k "cd /d %~dp0\frontend && npm run dev"

echo.
echo  -----------------------------------------------
echo   UI         http://localhost:3000
echo   Health     http://localhost:3000/health
echo   Airflow    http://localhost:8080   admin/admin
echo   MLflow     http://localhost:5000
echo   API docs   http://localhost:8000/docs
echo  -----------------------------------------------
echo.
echo  OpenRouter key must be set for RAG explanations.
echo  Trigger first ingest: airflow dags trigger dag_ingest_daily
echo.
pause
