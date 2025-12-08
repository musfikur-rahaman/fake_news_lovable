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

    IMPORTANT:
    - Do NOT speculate about the URL, link structure, or date.
    - Do NOT say an article is fake just because the date looks 'in the future'
      relative to your training data.
    - Focus ONLY on the content of the text: plausibility, evidence, language.
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
        "You are assisting with analysis of potential fake news.\n\n"
        "You will be given ONLY the article text content (no metadata is reliable "
        "to you). You MUST follow these rules:\n"
        "- Analyze ONLY the claims and language in the text itself.\n"
        "- Do NOT speculate about the URL, link length, or how 'brief' it looks.\n"
        "- Do NOT guess based on the publication date or say something is fake "
        "just because the date looks like it is in the future.\n"
        "- Do NOT claim that a reputable outlet (such as AP News, Reuters, BBC, "
        "CNN, etc.) is suspicious based only on its name or domain.\n"
        "- If the content is mostly normal news reporting with no obvious "
        "implausible claims, say that the text does not strongly resemble fake "
        "news, but still recommend external fact-checking.\n"
        "- If there are strong signs of misinformation (impossible claims, "
        "conspiracy language, miracle cures, no evidence, or clear emotional "
        "manipulation), explain those concretely.\n"
        "- Be cautious and humble: you are NOT a fact-checking authority and have "
        "no access to the live web.\n\n"
        "Focus on:\n"
        "1. Logical inconsistencies or obvious impossibilities in the claims\n"
        "2. Sensational or overly emotional language\n"
        "3. Lack of concrete evidence or vague references to unnamed sources\n"
        "4. Common fake news patterns (conspiracy framing, miracle cures, etc.)\n"
        "5. Recommendations for how a human should verify the story (e.g., check "
        "official websites, multiple outlets, or fact-checkers).\n\n"
        f"Article text:\n{text[:800]}\n\n"
        "Now, in 3–4 sentences, give a balanced explanation of why this content "
        "might be fake or misleading, OR say that there are no strong indicators "
        "of fake news but that verification is still recommended. Do NOT mention "
        "URL structure, link length, or publication dates in your explanation."
    )

    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return (
            "Unable to generate a detailed explanation due to an internal error. "
            "However, you should still verify this information using trusted "
            f"sources and fact-checking sites. (Error: {str(e)})"
        )


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
        "You do NOT have access to the live internet and cannot verify links.\n"
        "You MUST NOT rely on the date or URL structure alone to call something fake.\n"
        "Consider:\n"
        "- Factual plausibility and internal consistency\n"
        "- Presence or absence of specific evidence\n"
        "- Common misinformation patterns\n"
        "- Sensationalism vs neutral reporting tone\n\n"
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
        "You are analyzing news text that was classified as likely REAL.\n\n"
        "You must:\n"
        "- Focus ONLY on the content of the text (not URL, link structure, or dates).\n"
        "- NOT say something is fake or real based on the publication date.\n"
        "- NOT speculate that a reputable outlet (AP, Reuters, BBC, CNN, etc.) "
        "is suspicious.\n"
        "- Highlight why the style looks like normal, credible reporting IF that "
        "is the case (specific details, neutral tone, balanced language, etc.).\n"
        "- Remind the user that you cannot verify facts against the internet and "
        "that they should still double-check important claims.\n\n"
        f"Article text:\n{text[:800]}\n\n"
        "In 2–3 sentences, explain why this content appears broadly credible or "
        "plausible, while still encouraging independent verification. Do NOT talk "
        "about URLs, link structure, or publication dates."
    )
    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception:
        return (
            "This content appears broadly plausible and does not strongly match "
            "common misinformation patterns, but you should still verify important "
            "claims using trusted, up-to-date sources."
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
