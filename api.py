# api.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend_core import classify_news


app = FastAPI(title="Fake News Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str
    user_id: str | None = None


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
    return classify_news(req.text, user_id=req.user_id)
