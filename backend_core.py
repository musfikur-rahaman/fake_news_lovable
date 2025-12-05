# backend_core.py
import os
import time
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
from supabase import create_client, Client

# 👉 import your existing helpers
# from llmhelper import get_llm_explanation   # use whatever name you had
# from source_validator import check_source_reputation, analyze_url_characteristics
# from url_content_fetcher import is_url, extract_article_content, normalize_url

# from transformers import pipeline   # if you used HF pipelines

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_KEY missing from .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------------------------------------------------
# 1. Model loading – REUSE your existing model loading code here
# -------------------------------------------------------------------

_ENSEMBLE_MODELS = None
_MODEL_WEIGHTS = None

def load_ensemble_models():
    """
    TODO: copy your model initialization from your old Streamlit app.
    Example (pseudo):

        from transformers import pipeline

        models = {}
        weights = {}

        models["primary"] = pipeline(...)
        weights["primary"] = 0.7

        models["fallback"] = pipeline(...)
        weights["fallback"] = 0.3

    """
    global _ENSEMBLE_MODELS, _MODEL_WEIGHTS
    if _ENSEMBLE_MODELS is not None:
        return _ENSEMBLE_MODELS, _MODEL_WEIGHTS

    models = {}
    weights = {}

    # 🔴 TODO: COPY your real model loading here from old app.py
    # models["primary"] = ...
    # models["fallback"] = ...
    # weights["primary"] = 0.7
    # weights["fallback"] = 0.3

    _ENSEMBLE_MODELS = models
    _MODEL_WEIGHTS = weights
    return models, weights

# -------------------------------------------------------------------
# 2. Utility: label mapping / hallucination / fusion – REUSE
# -------------------------------------------------------------------

def map_label(label: str, model_name: str = "primary") -> str:
    """
    Map raw model label -> 'FAKE' or 'REAL'.
    TODO: copy your exact mapping logic here.
    """
    label_str = str(label).upper()

    # 🔴 TODO: if your existing logic is different, copy that instead.
    if model_name == "fallback":
        if "NEGATIVE" in label_str or "LABEL_0" in label_str:
            return "FAKE"
        elif "POSITIVE" in label_str or "LABEL_1" in label_str:
            return "REAL"
        else:
            return "REAL"

    if "FAKE" in label_str or "LABEL_1" in label_str:
        return "FAKE"
    elif "REAL" in label_str or "LABEL_0" in label_str:
        return "REAL"
    return "REAL"


def detect_hallucination_patterns(text: str) -> bool:
    """
    TODO: copy your existing hallucination detection from app.py.
    E.g., keywords like 'as an AI language model', 'I cannot browse the internet', etc.
    """
    # 🔴 PLACEHOLDER – replace with your real logic
    suspicious_phrases = [
        "as an ai language model",
        "i do not have access to real-time information",
        "cannot browse the internet",
    ]
    lower_text = text.lower()
    return any(p in lower_text for p in suspicious_phrases)


def fuse_predictions(
    ensemble_label: str,
    ensemble_confidence: float,
    halluc_flag: bool,
    source_reputation: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    TODO: copy your real fusion logic: how you combine model output,
    hallucination flag, and source reputation to get final label/confidence.
    """
    # 🔴 SIMPLE placeholder – replace with your logic
    final_label = ensemble_label
    final_confidence = ensemble_confidence

    if halluc_flag and ensemble_label == "REAL":
        final_label = "FAKE"
        final_confidence = max(final_confidence, 0.6)

    return {
        "label": final_label,
        "confidence": final_confidence,
    }

# -------------------------------------------------------------------
# 3. Ensemble classify – REUSE your logic
# -------------------------------------------------------------------

def ensemble_classify(text: str, source_url: Optional[str] = None):
    """
    Core classification – copy the logic from your Streamlit app:
    - run multiple models
    - aggregate their probabilities
    - check source reputation (if URL)
    - detect hallucination patterns
    - call fuse_predictions
    """
    models, weights = load_ensemble_models()

    # 🔴 TODO: copy your exact ensemble logic here from app.py.
    # Below is just a sketch to show the structure.

    ensemble_score = 0.0
    weight_sum = 0.0
    model_details: List[Dict[str, Any]] = []

    for name, clf in models.items():
        raw = clf(text)[0]  # {'label': 'FAKE', 'score': 0.87} for example
        mapped_label = map_label(raw["label"], model_name=name)
        score = raw["score"]
        w = weights.get(name, 1.0)

        if mapped_label == "FAKE":
            prob_fake = score
        else:
            prob_fake = 1.0 - score

        ensemble_score += w * prob_fake
        weight_sum += w

        model_details.append(
            {
                "model_name": name,
                "raw_label": raw["label"],
                "mapped_label": mapped_label,
                "score": score,
                "weight": w,
                "prob_fake": prob_fake,
            }
        )

    if weight_sum == 0:
        weight_sum = 1.0
    avg_prob_fake = ensemble_score / weight_sum

    ensemble_label = "FAKE" if avg_prob_fake >= 0.5 else "REAL"
    ensemble_confidence = avg_prob_fake if ensemble_label == "FAKE" else 1 - avg_prob_fake

    # source reputation
    source_reputation = None
    source_warnings: List[str] = []
    if source_url:
        # 🔴 uncomment when you copy your real helper
        # source_reputation = check_source_reputation(source_url)
        # source_warnings = analyze_url_characteristics(source_url)
        pass

    halluc_flag = detect_hallucination_patterns(text)

    fused = fuse_predictions(
        ensemble_label=ensemble_label,
        ensemble_confidence=ensemble_confidence,
        halluc_flag=halluc_flag,
        source_reputation=source_reputation,
    )

    return (
        fused["label"],
        fused["confidence"],
        halluc_flag,
        source_reputation,
        source_warnings,
        model_details,
    )

# -------------------------------------------------------------------
# 4. High-level classify function that Lovable will use via the API
# -------------------------------------------------------------------

def classify_news(input_text: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    - Detect URL vs. plain text
    - Fetch article content if URL
    - Run ensemble_classify
    - Call LLM for explanation
    - Save record to Supabase
    - Return JSON-ready dict
    """
    from url_content_fetcher import is_url, extract_article_content, normalize_url  # if exists
    from llmhelper import get_llm_explanation  # your function to get explanation

    raw_text = input_text.strip()
    normalized = normalize_url(raw_text) if raw_text else raw_text

    source_url = None
    article_text = raw_text
    fetch_error = None

    if normalized and is_url(normalized):
        source_url = normalized
        article_text, _, fetch_error = extract_article_content(source_url)
        if not article_text:
            article_text = raw_text

    (
        label,
        confidence,
        halluc_flag,
        source_rep,
        source_warnings,
        model_details,
    ) = ensemble_classify(article_text, source_url)

    explanation = get_llm_explanation(article_text, label)

    record = {
        "user_id": user_id,
        "input_text": raw_text,
        "resolved_text": article_text,
        "label": label,
        "confidence": confidence,
        "hallucination_flag": halluc_flag,
        "source_url": source_url,
        "source_reputation": source_rep,
        "source_warnings": source_warnings,
        "explanation": explanation,
        "model_details": model_details,
        "fetch_error": fetch_error,
        "created_at": int(time.time()),
    }

    # log in Supabase (ignore errors)
    try:
        supabase.table("news_analysis_history").insert(record).execute()
    except Exception:
        pass

    return record
