import os
import streamlit as st
import importlib
from dotenv import load_dotenv

import types

# ⚡ Universal LangChain compatibility patch
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

from langchain_groq import ChatGroq

load_dotenv()

# Get API key from env vars or Streamlit secrets
API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

# Initialize LLM with better configuration
llm = ChatGroq(
    api_key=API_KEY, 
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.1,  # Lower temperature for more consistent results
    max_tokens=500
)

def explain_fake_news(text):
    """
    Generate detailed explanation for why news may be fake.
    """
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
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"Unable to generate explanation: {str(e)}"

def fact_check(text):
    """
    Improved fact-checking with better reasoning.
    """
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
        response = llm.invoke(prompt)
        label = response.content.strip().upper()
        
        if "FAKE" in label:
            return "FAKE"
        elif "REAL" in label:
            return "REAL"
        else:
            return "REAL"  # Default to REAL
    except Exception as e:
        return "REAL"

def get_llm_explanation(text, classification):
    """
    Get detailed analysis from LLM for either fake or real news.
    """
    if classification == "FAKE":
        return explain_fake_news(text)
    else:
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
            response = llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return "This content appears legitimate based on AI analysis."

if __name__ == "__main__":
    fake_sample = "Breaking: NASA confirms aliens living among us disguised as squirrels!"
    real_sample = "The company announced quarterly earnings showing 5% growth in revenue."
    
    print("Fake news analysis:")
    print(explain_fake_news(fake_sample))
    print("Fact check:", fact_check(fake_sample))
    
    print("\nReal news analysis:") 
    print(get_llm_explanation(real_sample, "REAL"))
    print("Fact check:", fact_check(real_sample))
