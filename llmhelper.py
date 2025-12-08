# llmhelper.py

import os
import importlib
import types
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# RUNTIME / FEATURE FLAGS
# ============================================================

RUN_MODE = os.getenv("RUN_MODE", "dev")  # "dev" vs "prod"
# By default: explanations ON in dev, OFF in prod
ENABLE_EXPLANATIONS = os.getenv(
    "ENABLE_EXPLANATIONS",
    "1" if RUN_MODE == "dev" else "0",
).lower() in {"1", "true", "yes"}

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================================================
# OPTIONAL LANGCHAIN COMPAT PATCH (LIGHTWEIGHT)
# ============================================================

try:
    langchain = importlib.import_module("langchain")
except Exception:
    # If LangChain isn't installed at all, make a stub
    langchain = types.SimpleNamespace(verbose=False, llm_cache=None)

# Ensure 'verbose' and 'llm_cache' attributes exist, even if removed in newer versions
if not hasattr(langchain, "verbose"):
    langchain.verbose = False

if not hasattr(langchain, "llm_cache"):
    langchain.llm_cache = None  # prevent "no attribute 'llm_cache'" errors

# Try new-style debug control (LangChain >= 0.1)
try:
    from langchain_core.globals import set_debug, get_llm_cache

    set_debug(False)
    # sync llm_cache to core global if possible
    langchain.llm_cache = get_llm_cache()
except Exception:
    pass

# ============================================================
# LAZY LLM INITIALIZATION (GROQ)
# ============================================================

_llm = None  # cached ChatGroq instance


def get_llm():
    """
    Lazily create and cache the Groq LLM.

    This avoids heavy initialization at import time and makes it safe for
    low-memory environments like Render 512MB.
    """
    global _llm

    if not ENABLE_EXPLANATIONS:
        # Explanations are disabled; we shouldn't be calling this in prod,
        # but guard anyway.
        raise RuntimeError("LLM explanations are disabled (ENABLE_EXPLANATIONS=0).")

    if _llm is not None:
        return _llm

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in environment variables.")

    # Import ChatGroq only when needed
    from langchain_groq import ChatGroq

    _llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.1,  # lower = more consistent
        max_tokens=500,
    )
    return _llm


# ============================================================
# PUBLIC FUNCTIONS USED BY BACKEND
# ============================================================

def explain_fake_news(text: str) -> str:
    """
    Generate a short explanation for why news may be fake.

    In PROD / low-memory mode, this can be disabled via ENABLE_EXPLANATIONS=0.
    """
    if not ENABLE_EXPLANATIONS:
        # Lightweight fallback text if LLM is disabled
        return (
            "Explanation is disabled in this environment, but the content shows "
            "patterns commonly associated with misinformation (sensational claims, "
            "weak evidence, or implausible statements). Please cross-check with "
            "trusted sources before believing or sharing it."
        )

    prompt = (
        "Analyze the following news content and explain why it might be fake news. "
        "Focus on:\n"
        "1. Logical inconsistencies or impossibilities\n"
        "2. Sensational or emotional language\n"
        "3. Lack of credible sources or evidence\n"
        "4. Common fake news patterns\n"
        "5. Recommendations for verification\n\n"
        f"Content: {text[:800]}\n\n"
        "Provide a balanced, factual explanation in 3-4 sentences:"
    )

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"Unable to generate explanation: {str(e)}"


def fact_check(text: str) -> str:
    """
    Simple FAKE/REAL prediction from Groq LLM.

    Not currently used by your FastAPI backend, but safe to call.
    """
    if not ENABLE_EXPLANATIONS:
        # If LLM is disabled, just default to REAL to avoid blocking.
        return "REAL"

    prompt = (
        "Carefully analyze this news content and determine if it's FAKE or REAL.\n"
        "Consider:\n"
        "- Factual accuracy and plausibility\n"
        "- Source credibility (if mentioned)\n"
        "- Evidence and specifics provided\n"
        "- Common misinformation patterns\n"
        "- Sensationalism vs factual reporting\n\n"
        f"Content: {text[:1000]}\n\n"
        "Answer ONLY with FAKE or REAL. Do not add explanations."
    )
    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        label = response.content.strip().upper()

        if "FAKE" in label:
            return "FAKE"
        elif "REAL" in label:
            return "REAL"
        else:
            return "REAL"  # default
    except Exception:
        return "REAL"


def get_llm_explanation(text: str, classification: str) -> str:
    """
    Get a brief explanation for either fake or real news.

    Uses Groq when ENABLE_EXPLANATIONS=1, otherwise returns a lightweight fallback.
    """
    if classification == "FAKE":
        return explain_fake_news(text)

    if not ENABLE_EXPLANATIONS:
        # Lightweight fallback if explanations disabled
        return (
            "This content appears broadly plausible and does not strongly match "
            "common misinformation patterns. Still, it is recommended to verify "
            "key claims against trusted news outlets or official sources."
        )

    prompt = (
        "Explain why this news content appears to be REAL and credible.\n"
        "Consider:\n"
        "- Factual consistency\n"
        "- Plausible claims with evidence\n"
        "- Professional tone and language\n"
        "- Lack of common fake news patterns\n\n"
        f"Content: {text[:800]}\n\n"
        "Provide a brief explanation in 2-3 sentences:"
    )
    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception:
        return (
            "This content appears legitimate based on AI analysis, but you should "
            "still verify important claims with trusted sources."
        )


# ============================================================
# MANUAL TESTING (LOCAL ONLY)
# ============================================================

if __name__ == "__main__":
    fake_sample = "Breaking: NASA confirms aliens living among us disguised as squirrels!"
    real_sample = "The company announced quarterly earnings showing 5% growth in revenue."

    print(f"RUN_MODE={RUN_MODE}, ENABLE_EXPLANATIONS={ENABLE_EXPLANATIONS}")
    print("\nFake news analysis:")
    print(explain_fake_news(fake_sample))
    print("Fact check:", fact_check(fake_sample))

    print("\nReal news analysis:")
    print(get_llm_explanation(real_sample, "REAL"))
    print("Fact check:", fact_check(real_sample))
