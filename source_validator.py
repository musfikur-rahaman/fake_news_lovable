import re
from urllib.parse import urlparse

def extract_domain(url):
    """
    Extract clean domain from URL.
    Handles various URL formats and edge cases.
    """
    try:
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split('/')[0]
        
        # Remove www. prefix
        domain = domain.replace('www.', '').lower()
        
        # Remove port numbers if present
        domain = domain.split(':')[0]
        
        return domain if domain else None
    except Exception as e:
        return None

def check_source_reputation(url):
    """
    Check the reputation of a news source based on its domain.
    Returns a tuple: (reputation_level, emoji, description)
    """
    domain = extract_domain(url)
    
    if not domain:
        return "Invalid", "⚠️", "Invalid URL format. Please check and try again."
    
    # Comprehensive reputation database
    reputation_db = {
        # Highly Reliable - Established, fact-checked news organizations
        "bbc.com": ("Highly Reliable", "✅", "Publicly funded, editorially independent"),
        "reuters.com": ("Highly Reliable", "✅", "International news agency, fact-checked"),
        "apnews.com": ("Highly Reliable", "✅", "Associated Press - nonprofit news cooperative"),
        "nytimes.com": ("Highly Reliable", "✅", "Pulitzer Prize-winning journalism"),
        "npr.org": ("Highly Reliable", "✅", "Public radio, editorial standards"),
        "pbs.org": ("Highly Reliable", "✅", "Public broadcasting service"),
        "theguardian.com": ("Highly Reliable", "✅", "Fact-checked, editorial oversight"),
        "washingtonpost.com": ("Highly Reliable", "✅", "Major investigative journalism"),
        "wsj.com": ("Highly Reliable", "✅", "Wall Street Journal - business focus"),
        "economist.com": ("Highly Reliable", "✅", "International affairs analysis"),
        
        # Highly Reliable - Established, fact-checked news organizations
        "cnn.com": ("Highly Reliable", "✅", "Major news network with editorial standards"),
        "foxnews.com": ("Generally Reliable", "✔️", "Mainstream news, political bias noted"),
        "nbcnews.com": ("Generally Reliable", "✔️", "Major network news"),
        "cbsnews.com": ("Generally Reliable", "✔️", "Major network news"),
        "abcnews.go.com": ("Generally Reliable", "✔️", "Major network news"),
        "usatoday.com": ("Generally Reliable", "✔️", "National newspaper"),
        "time.com": ("Generally Reliable", "✔️", "News magazine"),
        "newsweek.com": ("Generally Reliable", "✔️", "News magazine"),
        
        # Mixed Reliability - Proceed with caution
        "huffpost.com": ("Mixed Reliability", "⚡", "Opinion-heavy, verify facts independently"),
        "dailymail.co.uk": ("Mixed Reliability", "⚡", "Tabloid journalism, sensationalism common"),
        "nypost.com": ("Mixed Reliability", "⚡", "Tabloid style, verify claims"),
        "buzzfeed.com": ("Mixed Reliability", "⚡", "Mix of news and entertainment"),
        "vice.com": ("Mixed Reliability", "⚡", "Alternative perspective, verify sources"),
        
        # Unreliable - High rate of misinformation
        "infowars.com": ("Unreliable", "❌", "Conspiracy theories, misinformation frequent"),
        "breitbart.com": ("Unreliable", "❌", "Extreme bias, fact-check all claims"),
        "naturalnews.com": ("Unreliable", "❌", "Pseudoscience, health misinformation"),
        "beforeitsnews.com": ("Unreliable", "❌", "Unverified user-generated content"),
        "worldnewsdailyreport.com": ("Unreliable", "❌", "Known for fabricated stories"),
        
        # Satire - Not real news
        "theonion.com": ("Satire", "😄", "Satirical news - intentionally fake for humor"),
        "thebeaverton.com": ("Satire", "😄", "Canadian satire site"),
        "clickhole.com": ("Satire", "😄", "Satirical clickbait parody"),
        "babylonbee.com": ("Satire", "😄", "Conservative satire site"),
        "newsthump.com": ("Satire", "😄", "British satire"),
        
        # Fact-checking organizations
        "snopes.com": ("Fact-Checker", "🔍", "Independent fact-checking"),
        "factcheck.org": ("Fact-Checker", "🔍", "Nonpartisan fact-checking"),
        "politifact.com": ("Fact-Checker", "🔍", "Political fact-checking"),
    }
    
    # Check if domain exists in database
    if domain in reputation_db:
        return reputation_db[domain]
    
    # Check for common patterns in unknown domains
    suspicious_patterns = [
        r'fake', r'hoax', r'conspiracy', r'leaked', r'exposed',
        r'truth', r'real', r'uncensored', r'insider', r'secret'
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, domain, re.IGNORECASE):
            return ("Potentially Unreliable", "⚠️", 
                    "Domain name suggests sensationalism. Verify independently.")
    
    # Unknown source
    return ("Unknown Source", "❓", 
            "No reputation data available. Verify through multiple sources.")

def get_source_score(reputation_level):
    """
    Convert reputation level to a numeric score (0-1) for hybrid classification.
    """
    scores = {
        "Highly Reliable": 0.1,      # Low fake news probability
        "Generally Reliable": 0.3,
        "Mixed Reliability": 0.5,
        "Potentially Unreliable": 0.7,
        "Unreliable": 0.9,           # High fake news probability
        "Satire": 1.0,               # Intentionally fake
        "Unknown Source": 0.5,       # Neutral
        "Fact-Checker": 0.0,         # Most reliable
        "Invalid": 0.5               # Cannot determine
    }
    return scores.get(reputation_level, 0.5)

def analyze_url_characteristics(url):
    """
    Analyze URL for suspicious characteristics.
    Returns list of warning flags.
    """
    warnings = []
    
    if not url:
        return warnings
    
    domain = extract_domain(url)
    if not domain:
        return ["Invalid URL format"]
    
    # Check for suspicious TLDs
    suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top']
    if any(domain.endswith(tld) for tld in suspicious_tlds):
        warnings.append("Suspicious domain extension")
    
    # Check for excessive hyphens (common in fake news sites)
    if domain.count('-') > 2:
        warnings.append("Unusual domain structure (multiple hyphens)")
    
    # Check for numbers in domain (sometimes suspicious)
    if re.search(r'\d{3,}', domain):
        warnings.append("Domain contains unusual number sequence")
    
    # Check for misspellings of popular sites
    popular_sites = ['google', 'facebook', 'twitter', 'cnn', 'bbc', 'nytimes']
    for site in popular_sites:
        if site in domain and domain != f"{site}.com":
            warnings.append(f"Possible typosquatting of {site}.com")
    
    return warnings

# Example usage and testing
if __name__ == "__main__":
    test_urls = [
        "https://www.bbc.com/news/world",
        "https://infowars.com/article",
        "theonion.com/news",
        "some-weird-news-site-123.tk",
        "invalid url",
        "https://fakennews.com"
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