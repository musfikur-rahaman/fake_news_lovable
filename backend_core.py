# backend_core.py

import os
import time
from typing import Optional, Dict, Any, List, Tuple

from dotenv import load_dotenv
from transformers import pipeline

try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None

from llmhelper import explain_fake_news
from source_validator import (
    check_source_reputation,
    analyze_url_characteristics,
)
from url_content_fetcher import is_url, extract_article_content, normalize_url


# ---------- ENV & SUPABASE SETUP (OPTIONAL) ----------

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Optional["Client"] = None
if SUPABASE_URL and SUPABASE_KEY and create_client is not None:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase client initialized")
    except Exception as e:
        print(f"⚠️ Could not initialize Supabase client: {e}")
        supabase = None
else:
    print("ℹ️ Supabase URL/KEY not set or supabase lib missing. Skipping DB logging.")


# ---------- ENSEMBLE MODEL CONFIGURATION ----------

MODEL_CONFIG = {
    "primary": "mrm8488/bert-tiny-finetuned-fake-news-detection",
    "fallback": "distilbert-base-uncased-finetuned-sst-2-english",
}

_ENSEMBLE_MODELS: Optional[Dict[str, Any]] = None
_MODEL_WEIGHTS: Optional[Dict[str, float]] = None


def load_ensemble_models() -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Lazily load and cache HuggingFace pipelines used for ensemble classification.
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
        # Emergency: try to at least load primary
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
    Map raw model labels to 'FAKE' or 'REAL'.
    Mirrors your Streamlit logic.
    """
    label_str = str(label).upper()

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
    else:
        return "REAL"


# ---------- PATTERN DETECTORS ----------

def detect_hallucination_patterns(text: str) -> bool:
    """
    Pattern-based detector for absurd / impossible claims.
    """
    t = text.lower()

    strong_indicators = [
        "microchip in vaccine", "5g caused", "flat earth", "alien body found",
        "alien mining", "flying pigs", "zombie outbreak", "immortality pill",
        "magic cure", "government hiding aliens", "time travel", "talking animals",
        "glowing blue river", "digitally altered", "fake photo", "deepfake",
        "photoshop", "edited image", "emotionally intelligent toaster",
        "ai president", "ai sworn in", "ai supreme court", "robot judge",
        "resurrected dinosaurs", "teleportation device", "weather control machine",
    ]
    if any(p in t for p in strong_indicators):
        return True

    contextual_indicators = [
        "breaking news", "shocking discovery", "they don't want you to know",
        "miracle cure", "secret revealed", "forbidden knowledge",
        "mainstream media won't tell you", "cover-up",
    ]
    if sum(1 for p in contextual_indicators if p in t) >= 2:
        return True

    logical_indicators = [
        "sworn into the u.s. supreme court", "nominated by ai",
        "machine president", "time traveler", "moon made of cheese",
        "aliens elected", "living statue", "eternal youth formula",
        "humans can fly without aid", "animals talking to humans",
    ]
    if any(p in t for p in logical_indicators):
        return True

    return False


def detect_sensational_or_fictional(text: str) -> bool:
    """
    Heuristic detector for obviously sensational / fictional content.
    E.g. 'AI robot gains citizenship', 'script written by cats'.
    """
    t = text.lower()

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
        "robot gains citizenship", "ai robot gains citizenship",
    ]

    if any(k in t for k in satire_keywords):
        return True
    if any(k in t for k in absurd_keywords):
        return True

    # Generic pattern: animals writing things
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
    - Boost / force FAKE for hallucination + sensational content
    - Adjust based on source reputation
    """
    label = ensemble_label
    confidence = ensemble_confidence

    if halluc_flag:
        confidence = min(confidence + 0.35, 0.99)
        label = "FAKE"

    if sensational_flag:
        if label == "REAL":
            label = "FAKE"
        confidence = max(confidence, 0.75)

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
    Run ensemble models, compute detailed per-model info, and aggregate.
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

    for model_key, clf in models.items():
        try:
            result = clf(text[:1000])[0]  # {'label': '...', 'score': ...}
            raw_label = str(result["label"])
            score = float(result["score"])
            mapped = map_label(raw_label, model_name=model_key)
            weight = float(model_weights.get(model_key, 0.1))

            if mapped == "FAKE":
                prob_fake = score
            else:
                prob_fake = 1.0 - score

            fake_score += weight * prob_fake
            real_score += weight * (1.0 - prob_fake)
            total_weight += weight

            # EXACT shape Lovable expects
            model_details.append(
                {
                    "model_name": model_key,
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

    halluc_flag = detect_hallucination_patterns(text)
    sensational_flag = detect_sensational_or_fictional(text)

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
    This implements the SAME RESPONSE SHAPE your API currently returns:
    - user_id
    - input_text
    - resolved_text
    - label
    - confidence
    - hallucination_flag
    - source_url
    - source_reputation
    - source_warnings
    - explanation
    - model_details
    - fetch_error
    - created_at
    Plus new field: sensational_flag (safe to add).
    """
    raw_input = input_text.strip()
    normalized_input = normalize_url(raw_input)

    resolved_text = raw_input
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
                resolved_text = raw_input
            else:
                print(f"✅ Article fetched: {article_title}")
                resolved_text = article_text
        except Exception as e:
            fetch_error = str(e)
            resolved_text = raw_input
    else:
        print("📄 Text detected; using raw input as resolved_text")
        source_url = None
        resolved_text = raw_input

    (
        label,
        confidence,
        halluc_flag,
        source_reputation,
        source_warnings,
        model_details,
        sensational_flag,
    ) = ensemble_classify(resolved_text, source_url)

    explanation: Optional[str] = None
    if label == "FAKE":
        try:
            explanation = explain_fake_news(resolved_text[:800])
        except Exception as e:
            print(f"❌ Explanation generation failed: {e}")
            explanation = None

    record: Dict[str, Any] = {
        "user_id": user_id,
        "input_text": raw_input,
        "resolved_text": resolved_text,
        "label": label,
        "confidence": confidence,
        "hallucination_flag": halluc_flag,
        "sensational_flag": sensational_flag,
        "source_url": source_url,
        "source_reputation": source_reputation,
        "source_warnings": source_warnings,
        "explanation": explanation,
        "model_details": model_details,
        "fetch_error": fetch_error,
        "created_at": int(time.time()),
    }

    # Optional: log to Supabase if configured
    if supabase is not None:
        try:
            supabase.table("news_analysis_history").insert(record).execute()
        except Exception as e:
            print(f"⚠️ Failed to log history in Supabase: {e}")

    return record
