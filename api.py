"""
FastAPI application to expose the SciSynth AI Research Assistant functionality.
"""

import os
import sys
import json
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add project root to Python path to allow module imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from agents.orchestrator import ResearchOrchestrator

# --- Pydantic Models for Request and Response ---


class ResearchQuery(BaseModel):
    """Request model for a research question."""

    question: str
    use_all_systems: bool = True


class HealthCheck(BaseModel):
    """Response model for health check."""

    status: str
    orchestrator_status: str


# --- FastAPI Application ---

app = FastAPI(
    title="SciSynth AI Research Assistant API",
    description="An API to access the multi-paradigm AI research assistant.",
    version="1.0.0",
)

# --- Global State ---
# This section handles loading models and data at startup.
# In a production environment, you might use a more robust solution for
# managing model lifecycle and state.

orchestrator = None


@app.on_event("startup")
def load_orchestrator():
    """Load the ResearchOrchestrator on application startup."""
    global orchestrator
    results_path = Path("results/all_papers_results.json")
    if not results_path.exists():
        # This is a critical error for the API, so we prevent startup.
        raise RuntimeError(
            f"FATAL: Results file not found at {results_path}. The API cannot start."
        )

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)

        # Initialize the orchestrator
        orchestrator = ResearchOrchestrator(all_results)
        print("✅ ResearchOrchestrator initialized successfully.")
    except Exception as e:
        # Log the error and prevent the app from starting in a broken state.
        print(f"❌ Error initializing ResearchOrchestrator: {e}")
        raise RuntimeError(f"Error during orchestrator initialization: {e}")


# --- API Endpoints ---


@app.get("/", tags=["Status"])
def read_root():
    """Root endpoint providing basic information."""
    return {"message": "Welcome to the SciSynth AI Research Assistant API"}


@app.get("/health", response_model=HealthCheck, tags=["Status"])
def health_check():
    """Health check endpoint to verify service status."""
    status = "ready" if orchestrator else "initializing"
    return {"status": "ok", "orchestrator_status": status}


@app.post("/analyze", tags=["AI Core"])
def analyze_research_question(query: ResearchQuery):
    """
    Perform a comprehensive research analysis on a given question.

    This is the main endpoint that leverages the full power of the AI system.
    """
    if not orchestrator:
        raise HTTPException(
            status_code=503,
            detail="Orchestrator is not available or still initializing.",
        )

    try:
        # The ask_question method conveniently handles both comprehensive and simple analyses
        results = orchestrator.ask_question(
            question=query.question, use_all_systems=query.use_all_systems
        )
        return results
    except Exception as e:
        # Catch potential errors from the analysis pipeline
        raise HTTPException(
            status_code=500, detail=f"An error occurred during analysis: {str(e)}"
        )


# --- Main entry point for running the API directly ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
