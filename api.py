# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from backend_core import classify_news

app = FastAPI(title="Fake News Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    text: str
    user_id: str | None = None

# NEW: root path so hitting / in browser shows something nice
@app.get("/")
def root():
    return {
        "message": "Fake News Detection API is running.",
        "endpoints": ["/health", "/api/analyze"],
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    result = classify_news(req.text, user_id=req.user_id)
    return result
