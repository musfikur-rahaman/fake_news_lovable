import re
from urllib.parse import urlparse

# ============================================================
# DOMAIN EXTRACTION
# ============================================================

def extract_domain(url):
    """
    Extract clean domain from URL.
    Handles various URL formats and edge cases.
    """
    try:
        if not url:
            return None

        url = url.strip()

        # Add protocol if missing
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split("/")[0]

        if not domain:
            return None

        # Remove www. prefix
        domain = domain.lower().replace("www.", "")

        # Remove port numbers if present
        domain = domain.split(":")[0]

        # Strip trailing dot
        domain = domain.rstrip(".")

        return domain if domain else None
    except Exception:
        return None


# ============================================================
# REPUTATION LOOKUP
# ============================================================

def check_source_reputation(url):
    """
    Check the reputation of a news source based on its domain.
    Returns a tuple: (reputation_level, emoji, description)
    """
    domain = extract_domain(url)

    if not domain:
        return "Invalid", "⚠️", "Invalid URL format. Please check and try again."

    # Core reputation database (curated, not exhaustive)
    reputation_db = {
        # ⭐⭐⭐ Highly Reliable – major fact-checked outlets
        "bbc.com": ("Highly Reliable", "✅", "Publicly funded, editorially independent"),
        "bbc.co.uk": ("Highly Reliable", "✅", "BBC UK service, regulated and fact-checked"),
        "reuters.com": ("Highly Reliable", "✅", "International news agency, fact-checked"),
        "apnews.com": ("Highly Reliable", "✅", "Associated Press – nonprofit news cooperative"),
        "nytimes.com": ("Highly Reliable", "✅", "Pulitzer Prize-winning journalism"),
        "npr.org": ("Highly Reliable", "✅", "Public radio with editorial standards"),
        "pbs.org": ("Highly Reliable", "✅", "Public broadcasting service"),
        "theguardian.com": ("Highly Reliable", "✅", "Fact-checked with editorial oversight"),
        "washingtonpost.com": ("Highly Reliable", "✅", "Major investigative journalism"),
        "wsj.com": ("Highly Reliable", "✅", "Wall Street Journal – business and finance"),
        "economist.com": ("Highly Reliable", "✅", "International affairs analysis"),
        "ft.com": ("Highly Reliable", "✅", "Financial Times – global business news"),
        "bloomberg.com": ("Highly Reliable", "✅", "Markets and business reporting"),
        "aljazeera.com": ("Highly Reliable", "✅", "Global news network"),
        "latimes.com": ("Highly Reliable", "✅", "Los Angeles Times – US regional daily"),
        "ap.org": ("Highly Reliable", "✅", "Associated Press main site"),

        # ✅ Generally Reliable – mainstream outlets (may have bias, still fact-checked)
        "cnn.com": ("Generally Reliable", "✔️", "Major news network with editorial standards"),
        "foxnews.com": ("Generally Reliable", "✔️", "Mainstream news, political bias noted"),
        "nbcnews.com": ("Generally Reliable", "✔️", "US network news"),
        "cbsnews.com": ("Generally Reliable", "✔️", "US network news"),
        "abcnews.go.com": ("Generally Reliable", "✔️", "US network news"),
        "usatoday.com": ("Generally Reliable", "✔️", "National US newspaper"),
        "time.com": ("Generally Reliable", "✔️", "News magazine"),
        "newsweek.com": ("Generally Reliable", "✔️", "News magazine"),
        "politico.com": ("Generally Reliable", "✔️", "Politics and policy coverage"),
        "forbes.com": ("Generally Reliable", "✔️", "Business and finance coverage"),
        "bostonglobe.com": ("Generally Reliable", "✔️", "Regional US newspaper"),
        "chicagotribune.com": ("Generally Reliable", "✔️", "Regional US newspaper"),
        "independent.co.uk": ("Generally Reliable", "✔️", "UK newspaper"),
        "telegraph.co.uk": ("Generally Reliable", "✔️", "UK newspaper"),
        "wsj.com": ("Generally Reliable", "✔️", "Business-focused newspaper"),

        # ⚡ Mixed Reliability – opinion heavy / tabloid / entertainment
        "huffpost.com": ("Mixed Reliability", "⚡", "Opinion-heavy, verify facts independently"),
        "dailymail.co.uk": ("Mixed Reliability", "⚡", "Tabloid style, sensational headlines"),
        "nypost.com": ("Mixed Reliability", "⚡", "Tabloid style, check claims"),
        "buzzfeed.com": ("Mixed Reliability", "⚡", "Mix of news and entertainment"),
        "vice.com": ("Mixed Reliability", "⚡", "Alternative perspectives, verify sources"),
        "mirror.co.uk": ("Mixed Reliability", "⚡", "Tabloid coverage, verify facts"),
        "the-sun.com": ("Mixed Reliability", "⚡", "Tabloid, heavy sensationalism"),
        "thesun.co.uk": ("Mixed Reliability", "⚡", "Tabloid, heavy sensationalism"),
        "rt.com": ("Mixed Reliability", "⚡", "State-backed outlet, verify information"),
        "sputniknews.com": ("Mixed Reliability", "⚡", "State-backed outlet, verify information"),

        # ❌ Unreliable – high rate of misinformation / conspiracies
        "infowars.com": ("Unreliable", "❌", "Conspiracy theories, misinformation frequent"),
        "breitbart.com": ("Unreliable", "❌", "Extreme bias, fact-check all claims"),
        "naturalnews.com": ("Unreliable", "❌", "Pseudoscience, health misinformation"),
        "beforeitsnews.com": ("Unreliable", "❌", "Unverified user-generated content"),
        "worldnewsdailyreport.com": ("Unreliable", "❌", "Known for fabricated stories"),
        "yournewswire.com": ("Unreliable", "❌", "Misinformation and fake stories"),
        "newswars.com": ("Unreliable", "❌", "Conspiracy-driven outlet"),
        "truthfrequencyradio.com": ("Unreliable", "❌", "Conspiracy / fringe content"),

        # 😄 Satire – intentionally fake for humor
        "theonion.com": ("Satire", "😄", "Satirical news, not factual"),
        "thebeaverton.com": ("Satire", "😄", "Canadian satire site"),
        "clickhole.com": ("Satire", "😄", "Satirical clickbait parody"),
        "babylonbee.com": ("Satire", "😄", "Conservative satire site"),
        "newsthump.com": ("Satire", "😄", "British satire site"),
        "waterfordwhispersnews.com": ("Satire", "😄", "Irish satire site"),
        "duffelblog.com": ("Satire", "😄", "Military satire"),

        # 🔍 Fact-checkers – very high trust for verifying claims
        "snopes.com": ("Fact-Checker", "🔍", "Independent fact-checking organization"),
        "factcheck.org": ("Fact-Checker", "🔍", "Nonpartisan fact-checking"),
        "politifact.com": ("Fact-Checker", "🔍", "US political fact-checking"),
        "fullfact.org": ("Fact-Checker", "🔍", "UK fact-checking charity"),
        "africacheck.org": ("Fact-Checker", "🔍", "Africa-focused fact checking"),
    }

    # Try exact domain lookup
    if domain in reputation_db:
        return reputation_db[domain]

    # Try reducing subdomains (e.g., edition.cnn.com → cnn.com)
    parts = domain.split(".")
    while len(parts) > 2:
        candidate = ".".join(parts[-2:])
        if candidate in reputation_db:
            return reputation_db[candidate]
        parts = parts[1:]

    # Blog / personal publishing platforms
    blog_platforms = ["substack.com", "medium.com", "wordpress.com", "blogspot.com"]
    if any(domain.endswith(p) for p in blog_platforms):
        return (
            "Mixed Reliability",
            "⚡",
            "Personal or independent publishing platform. Evaluate the specific author and sources.",
        )

    # If domain looks like a raw IP, treat as suspicious/unverified
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", domain):
        return (
            "Potentially Unreliable",
            "⚠️",
            "Content is hosted directly on an IP address. Verify source and intent carefully.",
        )

    # Suspicious / sensational patterns in unknown domains
    suspicious_patterns = [
        r"fake",
        r"hoax",
        r"conspiracy",
        r"leaked",
        r"exposed",
        r"whistleblower",
        r"patriot",
        r"truth",
        r"realnews",
        r"breakingnews",
        r"freedom",
        r"liberty",
        r"uncensored",
        r"insider",
        r"secret",
        r"alert",
        r"shock",
        r"viral",
        r"redpill",
    ]

    for pattern in suspicious_patterns:
        if re.search(pattern, domain, re.IGNORECASE):
            return (
                "Potentially Unreliable",
                "⚠️",
                "Domain name uses sensational or emotionally charged terms. Verify content with trusted sources.",
            )

    # Default – unknown but not obviously bad
    return (
        "Unknown Source",
        "❓",
        "No reputation data available. Cross-check important claims with multiple trusted sources.",
    )


# ============================================================
# NUMERIC SCORE
# ============================================================

def get_source_score(reputation_level):
    """
    Convert reputation level to a numeric score (0-1) for hybrid classification.
    Lower = more trustworthy, Higher = more likely to be fake.
    """
    scores = {
        "Highly Reliable": 0.1,       # Low fake news probability
        "Generally Reliable": 0.3,
        "Mixed Reliability": 0.4,
        "Potentially Unreliable": 0.7,
        "Unreliable": 0.9,            # High fake news probability
        "Satire": 1.0,                # Intentionally fake
        "Unknown Source": 0.5,        # Neutral / no info
        "Fact-Checker": 0.0,          # Most reliable
        "Invalid": 0.5,               # Cannot determine
    }
    return scores.get(reputation_level, 0.5)


# ============================================================
# URL CHARACTERISTIC ANALYSIS
# ============================================================

def analyze_url_characteristics(url):
    """
    Analyze URL for suspicious characteristics.
    Returns list of warning flags (strings).
    """
    warnings = []

    if not url:
        return warnings

    domain = extract_domain(url)
    if not domain:
        return ["Invalid URL format"]

    # Re-parse to look at path/query as well
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    path = parsed.path or ""
    query = parsed.query or ""

    # 1. Suspicious TLDs often used by low-quality sites or spam
    suspicious_tlds = [
        ".tk", ".ml", ".ga", ".cf", ".gq",
        ".xyz", ".top", ".click", ".info",
        ".biz", ".vip", ".loan", ".work",
    ]
    if any(domain.endswith(tld) for tld in suspicious_tlds):
        warnings.append("Suspicious or low-reputation domain extension")

    # 2. Excessive hyphens in domain (common in spam / deceptive sites)
    if domain.count("-") > 2:
        warnings.append("Unusual domain structure (multiple hyphens)")

    # 3. Long numeric sequences in domain
    if re.search(r"\d{3,}", domain):
        warnings.append("Domain contains long numeric sequence (may be less trustworthy)")

    # 4. Too many subdomains (e.g., a.b.c.d.example.com)
    if domain.count(".") > 3:
        warnings.append("Domain has many subdomains (could be trying to mimic another site)")

    # 5. Possible typosquatting of popular sites
    popular_sites = ["google", "facebook", "twitter", "cnn", "bbc", "nytimes", "youtube", "whatsapp"]
    for site in popular_sites:
        if site in domain and not domain.endswith(f"{site}.com") and domain != f"{site}.co.uk" and domain != f"{site}.org":
            warnings.append(f"Possible typosquatting or imitation of {site}")

    # 6. Clickbait or sensational wording in URL path
    clickbait_patterns = [
        "you-wont-believe",
        "you-will-not-believe",
        "shocking",
        "unbelievable",
        "what-happens-next",
        "what-happens-will",
        "top-10",
        "top10",
        "must-see",
        "must-read",
        "goes-viral",
        "goesviral",
        "secret-revealed",
        "the-truth-about",
    ]
    lower_path = path.lower()
    if any(p in lower_path for p in clickbait_patterns):
        warnings.append("URL path contains clickbait-style wording")

    # 7. Heavy tracking / marketing parameters
    if "utm_" in query or "ref=" in query or "affid=" in query or "affiliate=" in query:
        warnings.append("URL contains heavy tracking/affiliate parameters")

    # 8. Extremely long URL
    full_url_length = len(url)
    if full_url_length > 300:
        warnings.append("Very long URL (could be tracking-heavy or spammy)")

    return warnings


# ============================================================
# QUICK MANUAL TEST
# ============================================================

if __name__ == "__main__":
    test_urls = [
        "https://www.bbc.com/news/world",
        "https://infowars.com/article",
        "theonion.com/news",
        "some-weird-news-site-123.tk",
        "invalid url",
        "https://fakenews-alert-realtruth247.com",
        "https://medium.com/@randomuser/my-hot-take-on-politics",
        "http://192.168.1.10/news",
        "https://edition.cnn.com/politics/live-news/",
    ]

    print("Source Validation Examples:\n")
    for url in test_urls:
        level, emoji, desc = check_source_reputation(url)
        score = get_source_score(level)
        warnings = analyze_url_characteristics(url)

        print(f"URL: {url}")
        print(f"  {emoji} {level}: {desc}")
        print(f"  Fake News Score: {score}")
        if warnings:
            print(f"  Warnings: {', '.join(warnings)}")
        print()
