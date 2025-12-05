# backend_core.py
import os
import time
from typing import Optional, Dict, Any, List, Tuple

from dotenv import load_dotenv
from supabase import create_client, Client
from transformers import pipeline

from llmhelper import explain_fake_news
from source_validator import (
    check_source_reputation,
    get_source_score,          # not used directly yet, but imported if you need it
    analyze_url_characteristics,
    extract_domain,
)
from url_content_fetcher import is_url, extract_article_content, normalize_url

# ---------- LOAD ENV VARIABLES ----------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase URL or Key not found. Please set them in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------- ENSEMBLE MODEL CONFIGURATION ----------
MODEL_CONFIG = {
    "primary": "mrm8488/bert-tiny-finetuned-fake-news-detection",
    "fallback": "distilbert-base-uncased-finetuned-sst-2-english",
}

_ENSEMBLE_MODELS: Optional[Dict[str, Any]] = None
_MODEL_WEIGHTS: Optional[Dict[str, float]] = None


def load_ensemble_models() -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Lazily load and cache the HuggingFace pipelines used for ensemble classification.
    This mirrors the Streamlit version but without any UI calls.
    """
    global _ENSEMBLE_MODELS, _MODEL_WEIGHTS
    if _ENSEMBLE_MODELS is not None and _MODEL_WEIGHTS is not None:
        return _ENSEMBLE_MODELS, _MODEL_WEIGHTS

    models: Dict[str, Any] = {}
    model_weights: Dict[str, float] = {}

    # Primary fake-news model
    try:
        models["primary"] = pipeline(
            "text-classification",
            model=MODEL_CONFIG["primary"],
            device=-1,
            truncation=True,
            max_length=256,
        )
        model_weights["primary"] = 0.7
        print("✅ Loaded primary fake news model")
    except Exception as e:
        print(f"❌ Primary model failed: {e}")

    # Fallback sentiment model
    try:
        models["fallback"] = pipeline(
            "text-classification",
            model=MODEL_CONFIG["fallback"],
            device=-1,
            truncation=True,
            max_length=256,
        )
        model_weights["fallback"] = 0.3
        print("✅ Loaded fallback sentiment model")
    except Exception as e:
        print(f"⚠️ Fallback model failed: {e}")

    if not models:
        # As a last resort, try to at least load primary once more
        try:
            models["primary"] = pipeline(
                "text-classification",
                model=MODEL_CONFIG["primary"],
                device=-1,
                truncation=True,
                max_length=256,
            )
            model_weights["primary"] = 1.0
        except Exception as e:
            print(f"❌ All models failed to load: {e}")
            models = {}
            model_weights = {}

    _ENSEMBLE_MODELS = models
    _MODEL_WEIGHTS = model_weights
    return models, model_weights


def map_label(label: str, model_name: str = "primary") -> str:
    """
    Map raw model labels to 'FAKE' or 'REAL', matching your Streamlit logic.
    """
    label_str = str(label).upper()

    # Fallback sentiment model mapping
    if model_name == "fallback":
        if "NEGATIVE" in label_str or "LABEL_0" in label_str:
            return "FAKE"
        elif "POSITIVE" in label_str or "LABEL_1" in label_str:
            return "REAL"
        else:
            return "REAL"

    # Primary model mapping
    if "FAKE" in label_str or "LABEL_1" in label_str:
        return "FAKE"
    elif "REAL" in label_str or "LABEL_0" in label_str:
        return "REAL"
    else:
        return "REAL"


def detect_hallucination_patterns(text: str) -> bool:
    """
    Same pattern-based 'hallucination / absurd claim' detector from your Streamlit app.
    """
    text_lower = text.lower()

    strong_indicators = [
        "microchip in vaccine", "5g caused", "flat earth", "alien body found",
        "alien mining", "flying pigs", "zombie outbreak", "immortality pill",
        "magic cure", "government hiding aliens", "time travel", "talking animals",
        "glowing blue river", "digitally altered", "fake photo", "deepfake",
        "photoshop", "edited image", "emotionally intelligent toaster",
        "ai president", "ai sworn in", "ai supreme court", "robot judge",
        "resurrected dinosaurs", "teleportation device", "weather control machine",
    ]
    for phrase in strong_indicators:
        if phrase in text_lower:
            return True

    contextual_indicators = [
        "breaking news", "shocking discovery", "they don't want you to know",
        "miracle cure", "secret revealed", "forbidden knowledge",
        "mainstream media won't tell you", "cover-up",
    ]
    context_count = sum(1 for phrase in contextual_indicators if phrase in text_lower)
    if context_count >= 2:
        return True

    logical_indicators = [
        "sworn into the u.s. supreme court", "nominated by ai",
        "machine president", "time traveler", "moon made of cheese",
        "aliens elected", "living statue", "eternal youth formula",
        "humans can fly without aid", "animals talking to humans",
    ]
    for phrase in logical_indicators:
        if phrase in text_lower:
            return True

    return False


def fuse_predictions(
    ensemble_label: str,
    ensemble_confidence: float,
    halluc_flag: bool,
    source_reputation: Optional[Dict[str, Any]],
    model_details: List[Dict[str, Any]],
) -> Tuple[str, float]:
    """
    Apply your post-processing rules:
    - Boost confidence and force FAKE for hallucination patterns
    - Adjust based on source reputation
    """
    label = ensemble_label
    confidence = ensemble_confidence

    if halluc_flag:
        # Strong boost for impossible / absurd claims
        confidence = min(confidence + 0.35, 0.99)
        label = "FAKE"

    if source_reputation:
        rep_level = source_reputation.get("level", "")
        source_weight = 0.15
        if rep_level in ["Unreliable", "Satire"] and label == "FAKE":
            confidence = min(confidence + (0.2 * source_weight), 0.95)
        elif rep_level == "Highly Reliable" and label == "FAKE" and confidence < 0.75:
            confidence = confidence * (1 - source_weight)

    confidence = max(0.1, min(0.99, confidence))
    return label, confidence


def ensemble_classify(
    text: str,
    source_url: Optional[str] = None,
) -> Tuple[
    str,
    float,
    bool,
    Optional[Dict[str, Any]],
    List[str],
    List[Dict[str, Any]],
]:
    """
    Core ensemble classification logic adapted from your Streamlit app.
    Returns:
        final_label, final_confidence, halluc_flag,
        source_reputation, source_warnings, model_details
    """
    models, model_weights = load_ensemble_models()
    if not models:
        # Hard fallback
        return "REAL", 0.5, False, None, [], []

    predictions: List[str] = []
    confidence_scores: List[float] = []
    model_details: List[Dict[str, Any]] = []

    for model_name, model in models.items():
        try:
            # Limit length for performance
            result = model(text[:1000])[0]  # {'label': ..., 'score': ...}
            label = map_label(result["label"], model_name)
            score = float(result["score"])
            predictions.append(label)
            confidence_scores.append(score)
            model_details.append(
                {
                    "model": model_name,
                    "label": label,
                    "confidence": score,
                    "weight": model_weights.get(model_name, 0.1),
                }
            )
        except Exception as e:
            print(f"Model {model_name} failed: {e}")
            continue

    if not predictions:
        print("⚠️ All models failed, using fallback classification")
        return "REAL", 0.5, False, None, [], []

    fake_score = 0.0
    real_score = 0.0
    total_weight = 0.0

    # Weighted aggregation
    for i, (model_name, prediction) in enumerate(zip(models.keys(), predictions)):
        if model_name in model_weights and i < len(confidence_scores):
            weight = model_weights[model_name]
            confidence = confidence_scores[i]
            if prediction == "FAKE":
                fake_score += weight * confidence
            else:
                real_score += weight * confidence
            total_weight += weight

    if total_weight > 0:
        fake_score /= total_weight
        real_score /= total_weight

    ensemble_label = "FAKE" if fake_score > real_score else "REAL"
    ensemble_confidence = fake_score if ensemble_label == "FAKE" else real_score

    # Hallucination / absurdity patterns
    halluc_flag = detect_hallucination_patterns(text)

    # Source reputation
    source_reputation: Optional[Dict[str, Any]] = None
    source_warnings: List[str] = []
    if source_url and source_url.strip():
        rep_level, emoji, description = check_source_reputation(source_url)
        source_reputation = {
            "level": rep_level,
            "emoji": emoji,
            "description": description,
        }
        source_warnings = analyze_url_characteristics(source_url)

    final_label, final_confidence = fuse_predictions(
        ensemble_label,
        ensemble_confidence,
        halluc_flag,
        source_reputation,
        model_details,
    )

    return (
        final_label,
        final_confidence,
        halluc_flag,
        source_reputation,
        source_warnings,
        model_details,
    )


def classify_news(
    input_text: str,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High-level function used by the FastAPI endpoint.

    - Detects URL vs raw text
    - If URL, fetches and extracts article content + title
    - Runs ensemble_classify()
    - Optionally generates LLM explanation for FAKE news
    - Logs analysis to Supabase
    - Returns a JSON-serializable dict for the frontend
    """
    raw_input = input_text.strip()
    normalized_input = normalize_url(raw_input)

    article_text = raw_input
    article_title: Optional[str] = None
    source_url: Optional[str] = None
    fetch_error: Optional[str] = None

    # Decide if input is URL or text
    if normalized_input and is_url(normalized_input):
        source_url = normalized_input
        print(f"🔗 URL detected: {source_url}")
        try:
            article_text, article_title, error = extract_article_content(source_url)
            if error:
                fetch_error = error
                print(f"❌ URL fetch error: {error}")
                # fallback: use original text anyway
                article_text = raw_input
            else:
                print(f"✅ Article fetched: {article_title}")
        except Exception as e:
            fetch_error = str(e)
            article_text = raw_input
    else:
        print("📄 Text detected; using raw input as article_text")
        source_url = None
        article_text = raw_input

    # Run ensemble classification
    (
        label,
        confidence,
        halluc_flag,
        source_reputation,
        source_warnings,
        model_details,
    ) = ensemble_classify(article_text, source_url)

    # Optional LLM explanation (following your Streamlit logic: only for FAKE)
    explanation: Optional[str] = None
    if label == "FAKE":
        try:
            explanation = explain_fake_news(article_text[:800])
        except Exception as e:
            print(f"❌ Explanation generation failed: {e}")
            explanation = None

    # Build record (similar to your Streamlit history object)
    record: Dict[str, Any] = {
        "user_id": user_id,
        "news": article_text,
        "original_input": raw_input,
        "source_url": source_url,
        "article_title": article_title,
        "label": label,
        "confidence": confidence,
        "hallucination_flag": halluc_flag,
        "source_reputation": source_reputation,
        "source_warnings": source_warnings,
        "model_details": model_details,
        "fetch_error": fetch_error,
        "explanation": explanation,
        "timestamp": int(time.time()),
    }

    # Log to Supabase history table (adjust table name if needed)
    try:
        supabase.table("news_analysis_history").insert(record).execute()
    except Exception as e:
        print(f"⚠️ Failed to log history in Supabase: {e}")

    return record
