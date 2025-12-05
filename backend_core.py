# backend_core.py

import os
import time
from typing import Optional, Dict, Any, List, Tuple

from dotenv import load_dotenv

# Local transformers (for dev / laptop)
from transformers import pipeline

# Optional: Hugging Face Inference API (for Render)
try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

# Optional: Supabase logging
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

# ============================================================
# ENVIRONMENT & CLIENT SETUP
# ============================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

HF_API_KEY = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
USE_HF_INFERENCE = os.getenv("USE_HF_INFERENCE", "0").lower() in {"1", "true", "yes"}

supabase: Optional[Any] = None
if SUPABASE_URL and SUPABASE_KEY and create_client is not None:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase client initialized")
    except Exception as e:
        print(f"⚠️ Could not initialize Supabase client: {e}")
        supabase = None
else:
    print("ℹ️ Supabase URL/KEY not set or supabase lib missing. Skipping DB logging.")

hf_client: Optional[InferenceClient] = None
if USE_HF_INFERENCE and HF_API_KEY and InferenceClient is not None:
    try:
        hf_client = InferenceClient(token=HF_API_KEY)
        print("✅ Hugging Face InferenceClient initialized")
    except Exception as e:
        print(f"⚠️ Could not initialize HF InferenceClient: {e}")
        hf_client = None
elif USE_HF_INFERENCE:
    print("⚠️ USE_HF_INFERENCE is true but HF_API_KEY or InferenceClient is missing.")

# ============================================================
# ENSEMBLE MODEL CONFIGURATION
# ============================================================

MODEL_CONFIG = {
    # Small fake-news model
    "primary": "mrm8488/bert-tiny-finetuned-fake-news-detection",
    # Sentiment / style model
    "fallback": "distilbert-base-uncased-finetuned-sst-2-english",
}

_ENSEMBLE_MODELS: Optional[Dict[str, Any]] = None  # pipeline or HF model IDs
_MODEL_WEIGHTS: Optional[Dict[str, float]] = None


def load_ensemble_models() -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Lazily load and cache models/pipelines used for ensemble classification.

    Two modes:
      - Local: transformers.pipeline (good for local dev)
      - Remote: HuggingFace Inference API (good for Render)
    """
    global _ENSEMBLE_MODELS, _MODEL_WEIGHTS
    if _ENSEMBLE_MODELS is not None and _MODEL_WEIGHTS is not None:
        return _ENSEMBLE_MODELS, _MODEL_WEIGHTS

    models: Dict[str, Any] = {}
    model_weights: Dict[str, float] = {}

    # --------------------------------------------------------
    # Remote mode: Hugging Face Inference API
    # --------------------------------------------------------
    if USE_HF_INFERENCE and hf_client is not None:
        # In this mode, we just store the model_ids; inference happens via hf_client
        models["primary"] = MODEL_CONFIG["primary"]
        model_weights["primary"] = 0.7
        print(f"✅ Using HF Inference API for primary: {models['primary']}")

        models["fallback"] = MODEL_CONFIG["fallback"]
        model_weights["fallback"] = 0.3
        print(f"✅ Using HF Inference API for fallback: {models['fallback']}")

        _ENSEMBLE_MODELS = models
        _MODEL_WEIGHTS = model_weights
        return models, model_weights

    # --------------------------------------------------------
    # Local mode: transformers.pipeline
    # --------------------------------------------------------
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
        print("✅ Loaded primary fake news model (local pipeline)")
    except Exception as e:
        print(f"❌ Primary model failed (local pipeline): {e}")

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
        print("✅ Loaded fallback sentiment model (local pipeline)")
    except Exception as e:
        print(f"⚠️ Fallback model failed (local pipeline): {e}")

    if not models:
        print("❌ No models could be loaded.")
        _ENSEMBLE_MODELS = {}
        _MODEL_WEIGHTS = {}
        return {}, {}

    _ENSEMBLE_MODELS = models
    _MODEL_WEIGHTS = model_weights
    return models, model_weights


def _run_model_inference(model_key: str, model_obj: Any, text: str) -> Dict[str, Any]:
    """
    Unified inference helper.

    If using HF Inference API:
        model_obj is a model_id string, and we call hf_client.text_classification.
    Else:
        model_obj is a transformers pipeline and we call model_obj(text).
    Returns a dict: {"label": ..., "score": ...}
    """
    snippet = text[:1000]  # keep it short for safety

    # Remote mode
    if USE_HF_INFERENCE and hf_client is not None:
        model_id = str(model_obj)
        try:
            result = hf_client.text_classification(snippet, model=model_id)
            if isinstance(result, list) and len(result) > 0:
                r0 = result[0]
                return {"label": r0.get("label", "LABEL_0"), "score": float(r0.get("score", 0.5))}
        except Exception as e:
            print(f"HF Inference failed for {model_key} ({model_id}): {e}")
            raise

    # Local pipeline
    try:
        result = model_obj(snippet)[0]
        return {"label": result["label"], "score": float(result["score"])}
    except Exception as e:
        print(f"Local pipeline inference failed for {model_key}: {e}")
        raise


# ============================================================
# LABEL MAPPING
# ============================================================

def map_label(label: str, model_name: str = "primary") -> str:
    """
    Map raw model labels to 'FAKE' or 'REAL'.
    """
    label_str = str(label).upper()

    # DistilBERT sentiment-style mapping
    if model_name == "fallback":
        if "NEGATIVE" in label_str or "LABEL_0" in label_str:
            return "FAKE"
        elif "POSITIVE" in label_str or "LABEL_1" in label_str:
            return "REAL"
        else:
            return "REAL"

    # Primary fake-news model
    if "FAKE" in label_str or "LABEL_1" in label_str:
        return "FAKE"
    elif "REAL" in label_str or "LABEL_0" in label_str:
        return "REAL"
    else:
        return "REAL"


# ============================================================
# PATTERN-BASED DETECTORS
# ============================================================

def detect_hallucination_patterns(text: str) -> bool:
    """
    Pattern-based detector for absurd / impossible claims.
    This is a heuristic risk flag, not an absolute truth.
    """
    t = text.lower()

    strong_indicators = [
        "microchip in vaccine", "5g caused", "flat earth", "alien body found",
        "alien mining", "flying pigs", "zombie outbreak", "immortality pill",
        "magic cure", "government hiding aliens", "time travel", "talking animals",
        "glowing blue river", "digitally altered", "deepfake", "photoshop",
        "edited image", "emotionally intelligent toaster",
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
        "sworn into the u.s. supreme court", "machine president", "time traveler",
        "moon made of cheese", "aliens elected", "eternal youth formula",
        "humans can fly without aid", "animals talking to humans",
    ]
    if any(p in t for p in logical_indicators):
        return True

    return False


def detect_sensational_or_fictional(text: str) -> bool:
    """
    Heuristic detector for obviously sensational / fictional content.
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
        "talking animals", "teleportation device", "weather control machine",
        "robot gains citizenship", "ai robot gains citizenship",
    ]

    if any(k in t for k in satire_keywords):
        return True
    if any(k in t for k in absurd_keywords):
        return True

    animal_words = ["cat", "cats", "dog", "dogs", "hamster", "hamsters"]
    if ("wrote" in t or "written by" in t or "authored by" in t) and any(
        a in t for a in animal_words
    ):
        return True

    return False


# ============================================================
# FUSION & ENSEMBLE LOGIC
# ============================================================

def fuse_predictions(
    ensemble_label: str,
    ensemble_confidence: float,
    halluc_flag: bool,
    source_reputation: Optional[Dict[str, Any]],
    model_details: List[Dict[str, Any]],
    sensational_flag: bool,
) -> Tuple[str, float]:
    """
    Merge model predictions with heuristic detectors and source reputation.

    Key design:
      - Heuristic detectors (hallucination/sensational) are *risk flags*, not hard overrides.
      - For highly reliable sources (e.g., CNN), we are conservative about calling FAKE.
    """
    label = ensemble_label
    confidence = ensemble_confidence

    high_trust_source = (
        source_reputation is not None
        and source_reputation.get("level", "").lower() == "highly reliable"
    )

    # 1. Heuristic risk signals (soft influence)
    if halluc_flag or sensational_flag:
        if not high_trust_source:
            # Only push toward FAKE if we don't trust the source strongly
            if label == "REAL" and confidence < 0.75:
                label = "FAKE"
                confidence = max(confidence, 0.70)
            else:
                confidence = min(confidence + 0.10, 0.95)
        else:
            # For CNN / major outlets: treat as "flag this for review" but
            # do not flip REAL → FAKE on heuristics alone.
            confidence = min(confidence + 0.05, 0.90)

    # 2. Source reputation effects
    if source_reputation:
        rep_level = source_reputation.get("level", "")
        source_weight = 0.15

        if rep_level in ["Unreliable", "Satire"]:
            if label == "FAKE":
                confidence = min(confidence + 0.15, 0.99)
            elif label == "REAL" and confidence < 0.7:
                label = "FAKE"
                confidence = 0.75

        elif rep_level == "Highly Reliable":
            # Only accept FAKE if model is strongly confident
            if label == "FAKE" and confidence < 0.80:
                label = "REAL"
                confidence = max(confidence, 0.55)
            else:
                confidence = max(confidence, 0.60)

        else:
            # Neutral / mixed sources: small smoothing only
            confidence = confidence * (1.0 - source_weight) + source_weight * confidence

    # 3. Clamp confidence to [0.1, 0.99]
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
            result = _run_model_inference(model_key, clf, text)
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
            print(f"Model {model_key} failed during ensemble: {e}")
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


# ============================================================
# HIGH-LEVEL CLASSIFY FUNCTION FOR API
# ============================================================

def classify_news(
    input_text: str,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High-level function used by the FastAPI endpoint.
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
            print(f"❌ Exception while fetching article: {e}")
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
    # Only generate explanation if final label is FAKE
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

    # Optional DB logging
    if supabase is not None:
        try:
            supabase.table("news_analysis_history").insert(record).execute()
        except Exception as e:
            print(f"⚠️ Failed to log history in Supabase: {e}")

    return record
