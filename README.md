# NewsLogic — AI-Powered Fake News Detection

*(Suggested repo name: this repo is currently `fake_news_lovable`. Renaming it to `NewsLogic` gives the project a proper identity.)*

A full-stack fake news detection system: a FastAPI backend that classifies news text with a weighted model ensemble, scores the credibility of the source, and explains its verdict in plain language. Built for deployment on Render with a Lovable/React frontend.

## How it works

1. **Ensemble classification (70/30)** — two Hugging Face models vote on every article:
   - Primary: `mrm8488/bert-tiny-finetuned-fake-news-detection` (weight 0.7)
   - Fallback: `distilbert/distilbert-base-uncased-finetuned-sst-2-english` (weight 0.3)
   - Served via the Hugging Face Inference API in production (no GPU needed), with a local `transformers` pipeline option in dev.
2. **Source reputation check** — extracts the domain from a pasted URL, looks up its reputation, and scores URL characteristics (suspicious TLDs, URL patterns, etc.).
3. **Article extraction** — fetches and cleans the full text of a news URL so users can analyze a link, not just pasted text.
4. **LLM explanations** — a Groq-powered explainer translates the model verdict into a readable explanation of *why* the article looks real or fake.
5. **User accounts & history** — Supabase auth with per-user analysis history and a feedback/correction loop.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Service info |
| `/health` | GET | Health check (used by Render) |
| `/api/analyze` | POST | Classify news text |

Request:
```json
{ "text": "Article text or URL...", "user_id": "optional-user-id" }
```

## Project structure

- `api.py` — FastAPI app, routes, request/response models
- `backend_core.py` — ensemble loading, classification pipeline, orchestration
- `source_validator.py` — domain extraction, reputation lookup, URL scoring
- `url_content_fetcher.py` — article text extraction from URLs
- `llmhelper.py` — Groq LLM explanations
- `download_models.py` — pre-download models into the HF cache (dev/local use)
- `requirements.txt` — dependencies

## Quickstart

```bash
pip install -r requirements.txt
```

Create a `.env` file (never commit it):
```env
HF_API_KEY=your_huggingface_token
GROQ_API_KEY=your_groq_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
RUN_MODE=dev            # "dev" enables local pipelines + explanations
USE_HF_INFERENCE=true
ENABLE_EXPLANATIONS=1
```

Run:
```bash
uvicorn api:app --reload
```

## Deployment notes

- Designed for Render: `/health` endpoint included; CORS currently allows all origins — restrict `allow_origins` in `api.py` to your frontend domain before production use.
- In production the models run through the HF Inference API, keeping memory footprint small.

## Roadmap

- [ ] Consolidate the three prototype repos (`fakenews_detection`, `fake-news-detection-v1`) into this one
- [ ] Calibrate ensemble weights on a held-out benchmark set
- [ ] Add per-domain reputation data and caching

## Author

Musfikur Rahaman — PhD student, UA Little Rock
