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
    get_source_score,          # available if you want it later
    analyze_url_characteristics,
    extract_domain,
)
from url_content_fetcher import is_url, extract_article_content, normalize_url


# ---------- ENV & SUPABASE SETUP ----------

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
    """
    global _ENSEMBLE_MODELS, _MODEL_WEIGHTS
    if _ENSEMBLE_MODELS is not None and _MODEL_WEIGHTS is not None:
        return _ENSEMBLE_MODELS, _MODEL_WEIGHTS

    models: Dict[str, Any] = {}
    model_weights: Dict[str, float] = {}

    # Primary fake news model
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
        # Emergency: try to at least load primary as a single model
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


# ---------- PATTERN DETECTORS (HALLUCINATION / SENSATIONAL) ----------

def detect_hallucination_patterns(text: str) -> bool:
    """
    Pattern-based detector for absurd / impossible claims (from your Streamlit file).
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


def detect_sensational_or_fictional(text: str) -> bool:
    """
    Heuristic detector for obviously sensational, satirical, or fictional claims
    (e.g., 'movie script written by cats').
    """
    t = text.lower()

    # Direct satire / humor / fiction hints
    satire_keywords = [
        "satirical article", "satire piece", "satirical news",
        "this is satire", "parody article", "parody news",
        "humorous piece", "meant to be humorous", "joke article",
        "fictional story", "this story is fictional",
        "for entertainment purposes only",
    ]

    absurd_keywords = [
        "written by cats", "cats wrote the script", "script written by cats",
        "written by dogs", "authored by cats", "authored by dogs",
        "time traveler from the future",
        "shape-shifting reptilian", "lizard people",
        "aliens elected", "alien president",
        "zombie outbreak", "resurrected dinosaurs",
        "moon made of cheese", "immortality pill",
        "talking animals", "animals talking to humans",
        "teleportation device", "weather control machine",
    ]

    if any(k in t for k in satire_keywords):
        return True

    if any(k in t for k in absurd_keywords):
        return True

    # Generic pattern: animal "wrote" something
    animal_words = ["cat", "cats", "dog", "dogs", "hamster", "hamsters"]
    if "wrote" in t or "written by" in t or "authored by" in t:
        if any(a in t for a in animal_words):
            return True

    return False


# ---------- FUSION & ENSEMBLE LOGIC ----------

def fuse_predictions(
    ensemble_label: str,
    ensemble_confidence: float,
    halluc_flag: bool,
    source_reputation: Optional[Dict[str, Any]],
    model_details: List[Dict[str, Any]],
    sensational_flag: bool,
) -> Tuple[str, float]:
    """
    Apply post-processing rules:
    - Force / boost FAKE for hallucination and sensational content
    - Adjust based on source reputation
    """
    label = ensemble_label
    confidence = ensemble_confidence

    # 1) Impossible / absurd patterns
    if halluc_flag:
        confidence = min(confidence + 0.35, 0.99)
        label = "FAKE"

    # 2) Sensational / fictional content
    if sensational_flag:
        if label == "REAL":
            label = "FAKE"
        confidence = max(confidence, 0.75)

    # 3) Source reputation adjustments
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
    bool,
]:
    """
    Core ensemble classification logic.

    Returns:
        final_label, final_confidence, halluc_flag,
        source_reputation, source_warnings, model_details, sensational_flag
    """
    models, model_weights = load_ensemble_models()
    if not models:
        return "REAL", 0.5, False, None, [], [], False

    model_details: List[Dict[str, Any]] = []
    fake_score = 0.0
    real_score = 0.0
    total_weight = 0.0

    # Run each model and collect detailed info
    for model_key, clf in models.items():
        try:
            # e.g. {'label': 'LABEL_1', 'score': 0.87}
            result = clf(text[:1000])[0]
            raw_label = str(result["label"])
            score = float(result["score"])
            mapped = map_label(raw_label, model_name=model_key)
            weight = float(model_weights.get(model_key, 0.1))

            # Probability of FAKE based on mapped label
            if mapped == "FAKE":
                prob_fake = score
            else:
                prob_fake = 1.0 - score

            # Aggregate for ensemble
            fake_score += weight * prob_fake
            real_score += weight * (1.0 - prob_fake)
            total_weight += weight

            # 👇 This is exactly the structure Lovable expects
            model_details.append(
                {
                    "model_name": model_key,      # you can rename more nicely if you want
                    "raw_label": raw_label,
                    "mapped_label": mapped,
                    "score": score,
                    "weight": weight,
                    "prob_fake": prob_fake,
                }
            )

        except Exception as e:
            print(f"Model {model_key} failed: {e}")
            continue

    if total_weight <= 0:
        print("⚠️ All models failed, using fallback classification")
        return "REAL", 0.5, False, None, [], [], False

    fake_score /= total_weight
    real_score /= total_weight

    ensemble_label = "FAKE" if fake_score > real_score else "REAL"
    ensemble_confidence = fake_score if ensemble_label == "FAKE" else real_score

    # Hallucination / absurd patterns
    halluc_flag = detect_hallucination_patterns(text)

    # Sensational / fictional patterns
    sensational_flag = detect_sensational_or_fictional(text)

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
        sensational_flag,
    )

    return (
        final_label,
        final_confidence,
        halluc_flag,
        source_reputation,
        source_warnings,
        model_details,
        sensational_flag,
    )


# ---------- HIGH-LEVEL CLASSIFY FUNCTION FOR API ----------

def classify_news(
    input_text: str,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High-level function used by the FastAPI endpoint.

    - Detects URL vs raw text
    - If URL, fetches and extracts article content + title
    - Runs ensemble_classify()
    - Generates LLM explanation for FAKE news
    - Logs analysis to Supabase
    - Returns JSON-serializable dict
    """
    raw_input = input_text.strip()
    normalized_input = normalize_url(raw_input)

    article_text = raw_input
    article_title: Optional[str] = None
    source_url: Optional[str] = None
    fetch_error: Optional[str] = None

    # Detect URL vs text
    if normalized_input and is_url(normalized_input):
        source_url = normalized_input
        print(f"🔗 URL detected: {source_url}")
        try:
            article_text, article_title, error = extract_article_content(source_url)
            if error:
                fetch_error = error
                print(f"❌ URL fetch error: {error}")
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

    (
        label,
        confidence,
        halluc_flag,
        source_reputation,
        source_warnings,
        model_details,
        sensational_flag,
    ) = ensemble_classify(article_text, source_url)

    # LLM explanation for FAKE news
    explanation: Optional[str] = None
    if label == "FAKE":
        try:
            explanation = explain_fake_news(article_text[:800])
        except Exception as e:
            print(f"❌ Explanation generation failed: {e}")
            explanation = None

    record: Dict[str, Any] = {
        "user_id": user_id,
        "news": article_text,
        "original_input": raw_input,
        "source_url": source_url,
        "article_title": article_title,
        "label": label,
        "confidence": confidence,
        "hallucination_flag": halluc_flag,
        "sensational_flag": sensational_flag,
        "source_reputation": source_reputation,
        "source_warnings": source_warnings,
        "model_details": model_details,  # 👈 now in the format Lovable wants
        "fetch_error": fetch_error,
        "explanation": explanation,
        "timestamp": int(time.time()),
    }

    # Adjust table name if your Supabase table is different
    try:
        supabase.table("news_analysis_history").insert(record).execute()
    except Exception as e:
        print(f"⚠️ Failed to log history in Supabase: {e}")

    return record
