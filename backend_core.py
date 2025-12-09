# backend_core.py

import os
import time
from typing import Optional, Dict, Any, List, Tuple

from dotenv import load_dotenv

# ============================================================
# ENVIRONMENT & MODE SETUP
# ============================================================

# Load .env for local dev; in Render, env vars are already set
load_dotenv()

RUN_MODE = os.getenv("RUN_MODE", "dev")  # "dev" (local) vs "prod"/others (Render, etc.)

# Explanations ON by default in dev, OFF by default in prod
ENABLE_EXPLANATIONS = os.getenv(
    "ENABLE_EXPLANATIONS",
    "1" if RUN_MODE == "dev" else "0",
).lower() in {"1", "true", "yes"}

# Supabase table (fixed to analysis_history by default)
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "analysis_history")

# ============================================================
# CONDITIONAL IMPORTS
# ============================================================

# Local transformers (for dev / laptop ONLY)
if RUN_MODE == "dev":
    try:
        from transformers import pipeline
    except ImportError:
        pipeline = None
        print("⚠️ transformers not installed; local pipeline mode disabled.")
else:
    pipeline = None  # Never use local pipelines in production to avoid OOM

# Optional: Hugging Face Inference API (for Render / remote inference)
InferenceClient = None
try:
    # Newer huggingface_hub versions (>= 0.20)
    from huggingface_hub.inference import InferenceClient  # type: ignore
except Exception:
    try:
        # Older huggingface_hub versions (< 0.20)
        from huggingface_hub import InferenceClient  # type: ignore
    except Exception:
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

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

HF_API_KEY = (
    os.getenv("HF_API_KEY")
    or os.getenv("HF_API_TOKEN")
    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
USE_HF_INFERENCE = os.getenv("USE_HF_INFERENCE", "0").lower() in {"1", "true", "yes"}

print(
    f"App starting with RUN_MODE={RUN_MODE}, "
    f"USE_HF_INFERENCE={USE_HF_INFERENCE}, "
    f"ENABLE_EXPLANATIONS={ENABLE_EXPLANATIONS}"
)

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

# Primary = fake-news model, fallback = sentiment model
# 🔄 UPDATED:
#   - primary: DeBERTa-v3-base (fake news classifier / base encoder)
#   - fallback: Twitter-RoBERTa-base sentiment (tone / polarity)
MODEL_CONFIG = {
    # Fake news classifier (now DeBERTa-v3-base)
    "primary": "microsoft/deberta-v3-base",
    # Sentiment / tone classifier (Twitter-RoBERTa sentiment)
    "fallback": "cardiffnlp/twitter-roberta-base-sentiment-latest",
}

_ENSEMBLE_MODELS: Optional[Dict[str, Any]] = None  # pipeline or HF model IDs
_MODEL_WEIGHTS: Optional[Dict[str, float]] = None


def load_ensemble_models() -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Lazily load and cache models/pipelines used for ensemble classification.

    Modes:
      - Remote: HuggingFace Inference API (preferred for Render / low-RAM)
      - Local: transformers.pipeline (ONLY in RUN_MODE=dev)
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
    # Local mode: transformers.pipeline (dev only)
    # --------------------------------------------------------
    if RUN_MODE != "dev":
        print("⚠️ RUN_MODE != 'dev' and USE_HF_INFERENCE is false; no local models will be loaded.")
        _ENSEMBLE_MODELS = {}
        _MODEL_WEIGHTS = {}
        return {}, {}

    if pipeline is None:
        print("⚠️ transformers.pipeline is not available; cannot load local models.")
        _ENSEMBLE_MODELS = {}
        _MODEL_WEIGHTS = {}
        return {}, {}

    # Primary fake news model (local dev only)
    try:
        models["primary"] = pipeline(
            "text-classification",
            model=MODEL_CONFIG["primary"],
            device=-1,
            truncation=True,
            max_length=256,
        )
        model_weights["primary"] = 0.7
        print("✅ Loaded primary fake news model (local pipeline, DeBERTa-v3-base)")
    except Exception as e:
        print(f"❌ Primary model failed (local pipeline): {e}")

    # Fallback sentiment model (local dev only)
    try:
        models["fallback"] = pipeline(
            "text-classification",
            model=MODEL_CONFIG["fallback"],
            device=-1,
            truncation=True,
            max_length=256,
        )
        model_weights["fallback"] = 0.3
        print("✅ Loaded fallback sentiment model (local pipeline, Twitter-RoBERTa sentiment)")
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
                return {
                    "label": r0.get("label", "LABEL_0"),
                    "score": float(r0.get("score", 0.5)),
                }
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

    - primary  = fake-news classifier (DeBERTa-v3-base used as encoder; we assume
                 LABEL_1 or FAKE-ish label means 'fake', LABEL_0 or REAL-ish means 'real').
    - fallback = sentiment model (Twitter-RoBERTa sentiment: NEGATIVE ~ FAKE risk,
                 NEUTRAL/POSITIVE ~ REAL-leaning).
    """
    label_str = str(label).upper()

    # Fallback sentiment-style mapping (Twitter-RoBERTa: NEGATIVE / NEUTRAL / POSITIVE)
    if model_name == "fallback":
        if "NEGATIVE" in label_str or "LABEL_0" in label_str:
            return "FAKE"
        elif "POSITIVE" in label_str or "LABEL_2" in label_str:
            return "REAL"
        elif "NEUTRAL" in label_str or "LABEL_1" in label_str:
            # Treat neutral tone as slightly more likely to be REAL than FAKE
            return "REAL"
        else:
            # Safe default
            return "REAL"

    # Primary fake-news model (DeBERTa encoder fine-tuned or used via generic head)
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
    Pattern-based detector for absurd / impossible / highly implausible claims.
    This is a heuristic risk flag, not a ground-truth detector.
    Returns True if the text strongly looks like hallucination / fake content.
    """
    t = text.lower()

    # 1. Physically / scientifically impossible claims
    impossible_science = [
        "humans can fly without aid",
        "humans can fly without wings",
        "breathing underwater without equipment",
        "breathe underwater without equipment",
        "perpetual motion machine",
        "free energy device",
        "infinite energy machine",
        "defies the laws of physics",
        "breaks the laws of physics",
        "gravity turned off",
        "gravity reversed",
        "sun stopped in the sky",
        "the sun stopped moving",
        "the moon disappeared",
        "moon made of cheese",
        "earth stopped rotating",
        "earth stopped spinning",
        "faster than light travel",
        "ftl travel",
        "teleportation device",
        "instant teleportation",
        "time travel machine",
        "time-travel machine",
        "time traveler from the future",
        "time traveller from the future",
        "time traveler from the past",
        "time traveller from the past",
        "resurrected dinosaurs",
        "ancient dinosaurs roaming cities",
        "telekinesis is now real",
        "telepathy is now proven",
        "controlling weather with a machine",
        "weather control machine",
        "controlling earthquakes with a device",
    ]

    # 2. Pseudoscientific / miracle cure claims
    miracle_health = [
        "cures all diseases",
        "cure for every disease",
        "instant cure for cancer",
        "cancer cured overnight",
        "cancer cured in one day",
        "no side effects and cures everything",
        "miracle cure",
        "secret cure doctors don't want you to know",
        "secret cure doctors dont want you to know",
        "heals any illness instantly",
        "heals every illness instantly",
        "reverse aging completely",
        "stop aging completely",
        "immortality pill",
        "live forever with this pill",
        "live forever thanks to this",
        "100% guaranteed weight loss overnight",
        "lose 50 pounds in a week with no effort",
        "reverse death",
        "bring the dead back to life",
        "raises the dead",
        "superfood that cures everything",
        "magic herb that cures all",
        "quantum healing for all diseases",
    ]

    # 3. Extreme conspiracy / grand control narratives
    extreme_conspiracies = [
        "global elite controls every government",
        "one secret group controls the world",
        "shadow government controls everything",
        "secret world government",
        "world is secretly ruled by lizard people",
        "world is secretly ruled by reptilians",
        "shape-shifting reptilian",
        "shape shifting reptilian",
        "lizard people running the government",
        "vaccines contain microchips",
        "vaccine contains a microchip",
        "microchip in vaccine",
        "5g caused the virus",
        "5g caused covid",
        "chemtrails are mind control",
        "chemtrails control the population",
        "moon landing was filmed in a studio",
        "all news is fake except this",
        "mainstream media always lies",
        "government hiding aliens among us",
        "government hiding aliens",
        "aliens living among us disguised as humans",
        "aliens living among us disguised as animals",
        "flat earth proof",
        "earth is flat and nasa is lying",
        "nasa is hiding the truth about earth",
        "massive global cover-up",
        "they don't want you to know the truth",
        "they dont want you to know the truth",
        "truth they don't want you to know",
        "truth they dont want you to know",
        "secret society controls reality",
        "secret society controls everything",
        "world leaders are actually androids",
    ]

    # 4. Supernatural / magical power claims in a news tone
    supernatural_claims = [
        "real wizard discovered",
        "real witch discovered",
        "witch casts spell on entire city",
        "spell cast over an entire country",
        "vampires confirmed real",
        "werewolves confirmed real",
        "zombie outbreak",
        "zombie apocalypse has started",
        "talking animals announce",
        "animals suddenly learned to talk",
        "ghost caught on official camera",
        "ghost elected to office",
        "demon caught on tape",
        "angel spotted giving interviews",
        "magic wand that changes reality",
        "magic ritual broadcast on live tv",
        "portal to another dimension opened",
        "doorway to another universe opened",
    ]

    # 5. Obviously fictional tech / AI roles framed as current reality
    absurd_tech = [
        "ai president of the united states",
        "ai sworn in as president",
        "ai sworn in as the president",
        "ai sworn in as prime minister",
        "ai is now the king",
        "robot judge in the supreme court",
        "robot judge in the u.s. supreme court",
        "ai supreme court justice",
        "ai wins human election",
        "robot elected as president",
        "robot elected as prime minister",
        "first cyborg president",
        "self-aware toaster takes over the world",
        "emotionally intelligent toaster",
        "sentient fridge gives political speeches",
        "ai takes full control of all governments",
        "single computer now controls all money",
        "all human jobs replaced overnight by ai",
    ]

    # 6. Totally absurd / comedic imagery
    absurd_imagery = [
        "flying pigs spotted",
        "pigs flying over the city",
        "cows flying through the sky",
        "fish walking on land and giving interviews",
        "cats wrote the script",
        "script written by cats",
        "governed entirely by hamsters",
        "hamsters now run the country",
        "penguins discovered living in the sahara desert",
        "desert suddenly turned into chocolate",
        "sky turned into cheese",
        "trees suddenly started walking",
    ]

    # 7. Strong "obviously edited" image / deepfake hints
    edited_media = [
        "clearly photoshopped image",
        "digitally altered image",
        "digitally altered photo",
        "deepfake video",
        "obvious deepfake",
        "this image is clearly edited",
        "obvious photoshop",
        "blatantly edited screenshot",
        "fake screenshot circulating online",
    ]

    # 8. "Too good to be true" unlimited deals framed as real news
    impossible_offers = [
        "every citizen will receive unlimited money",
        "government will pay everyone a million dollars",
        "bank gives free money to everyone",
        "no work required and you get rich",
        "everyone's debt will be erased overnight",
        "everyones debt will be erased overnight",
        "worldwide student loans cancelled instantly",
    ]

    for group in [
        impossible_science,
        miracle_health,
        supernatural_claims,
        absurd_tech,
        absurd_imagery,
        edited_media,
        impossible_offers,
    ]:
        if any(p in t for p in group):
            return True

    if any(p in t for p in extreme_conspiracies):
        return True

    soft_triggers = [
        "this proves the earth is flat",
        "undeniable proof that everything is fake",
        "everyone has been lied to for centuries",
        "nothing you know is real",
        "everything you know is a lie",
    ]
    if any(p in t for p in soft_triggers):
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

    Updated policy:
      - For HIGH-TRUST sources (Highly Reliable, Generally Reliable, Fact-Checker),
        we are VERY conservative about calling FAKE.
      - Heuristic detectors (hallucination/sensational) are risk flags, not
        hard overrides, especially for high-trust domains.
    """
    label = ensemble_label
    confidence = ensemble_confidence

    rep_level_raw = source_reputation.get("level") if source_reputation else None
    rep_level = (rep_level_raw or "").lower()

    is_highly_reliable = rep_level == "highly reliable"
    is_generally_reliable = rep_level == "generally reliable"
    is_fact_checker = rep_level == "fact-checker"

    high_trust_source = is_highly_reliable or is_generally_reliable or is_fact_checker

    # ------------------------------
    # 1. Heuristic risk signals
    # ------------------------------
    if halluc_flag or sensational_flag:
        if not high_trust_source:
            # Only push toward FAKE if we don't trust the source strongly
            if label == "REAL" and confidence < 0.75:
                label = "FAKE"
                confidence = max(confidence, 0.70)
            else:
                confidence = min(confidence + 0.10, 0.95)
        else:
            # For high-trust sources, heuristics only slightly adjust confidence.
            # They do NOT flip REAL → FAKE on their own.
            confidence = min(confidence + 0.05, 0.90)

    # ------------------------------
    # 2. Source reputation effects
    # ------------------------------
    if source_reputation:
        # LOW / UNKNOWN trust sources behave similar to previous logic
        if rep_level in ["unreliable", "satire"]:
            if label == "FAKE":
                confidence = min(confidence + 0.15, 0.99)
            elif label == "REAL" and confidence < 0.7:
                label = "FAKE"
                confidence = 0.75

        elif rep_level == "mixed reliability":
            # Slightly nudge confidence but don't flip by itself
            confidence = min(max(confidence, 0.5), 0.95)

        # HIGH TRUST HANDLING (strong guardrails)
        elif is_fact_checker:
            # Fact-checkers are extremely unlikely to publish fake content.
            # In this system we NEVER output FAKE for fact-checking domains.
            label = "REAL"
            confidence = max(confidence, 0.80)

        elif is_highly_reliable:
            # Highly reliable (BBC, Reuters, AP, NYT, etc.)
            if label == "FAKE":
                # Only keep FAKE if VERY high confidence + heuristics triggered
                if (halluc_flag or sensational_flag) and confidence >= 0.98:
                    confidence = max(confidence, 0.90)
                else:
                    label = "REAL"
                    confidence = max(confidence, 0.75)
            else:
                # If labeled REAL, ensure decent minimum confidence
                confidence = max(confidence, 0.70)

        elif is_generally_reliable:
            # Mainstream but possibly biased (CNN, Fox, etc.)
            if label == "FAKE":
                # Require extremely high confidence + heuristics
                if (halluc_flag or sensational_flag) and confidence >= 0.99:
                    confidence = max(confidence, 0.92)
                else:
                    label = "REAL"
                    confidence = max(confidence, 0.70)
            else:
                confidence = max(confidence, 0.65)

        else:
            # Neutral / unknown sources: light smoothing only
            source_weight = 0.15
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

    resolved_text = raw_input
    source_url: Optional[str] = None
    fetch_error: Optional[str] = None

    # Detect URL vs text using the RAW input first
    if is_url(raw_input):
        source_url = normalize_url(raw_input)
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
    # Always generate some explanation for FAKE; llmhelper decides whether to call LLM or fallback
    if label == "FAKE":
        try:
            print("🧠 Generating explanation for FAKE prediction...")
            explanation = explain_fake_news(resolved_text[:800])
            print("🧠 Explanation generated.")
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
            supabase.table(SUPABASE_TABLE).insert(record).execute()
        except Exception as e:
            print(f"⚠️ Failed to log history in Supabase ({SUPABASE_TABLE}): {e}")

    return record
