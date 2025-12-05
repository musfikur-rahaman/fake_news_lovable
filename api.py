# api.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from backend_core import classify_news


app = FastAPI(
    title="Fake News Detection API",
    description="Backend for the AI-Powered Fake News Detection System (HF Inference API + Supabase).",
    version="1.0.0",
)

# ---------- CORS (you can tighten this later) ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # TODO: restrict to your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Request / Response Models ----------

class AnalyzeRequest(BaseModel):
    text: str
    user_id: Optional[str] = None


@app.get("/")
def root():
    return {
        "message": "Fake News Detection API is running.",
        "endpoints": ["/health", "/api/analyze"],
    }


@app.get("/health")
def health():
    # Simple health check for Render
    return {"status": "ok"}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    """
    Main analysis endpoint.

    Request body:
    {
      "text": "...",
      "user_id": "optional-user-id"
    }

    Response body matches classify_news(...) record from backend_core.py.
    """
    result = classify_news(req.text, user_id=req.user_id)
    return result
