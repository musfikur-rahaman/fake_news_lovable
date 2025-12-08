import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse

# ============================================================
# URL DETECTION
# ============================================================

def is_url(text: str) -> bool:
    """
    Detect if the input text contains a URL.
    - Handles full URLs (http/https)
    - Handles 'www.' style
    - Handles bare domains like 'nytimes.com/article'
    """
    if not text:
        return False

    text = text.strip()

    # If there's whitespace, scan token by token so we don't
    # treat a whole long sentence as the URL itself.
    tokens = text.split()
    candidates = tokens if len(tokens) > 1 else [text]

    url_regex = re.compile(
        r"""(?ix)                             # ignore case, verbose
        \b(
            https?://[^\s]+                  # http(s)://...
            |
            www\.[^\s]+                      # www....
            |
            [a-z0-9.-]+\.
            (com|org|net|edu|gov|uk|co|io|ai|news|info|biz|de|fr|ca|au)
            (/[^\s]*)?                       # optional path
        )
        """,
    )

    for cand in candidates:
        if url_regex.search(cand):
            return True

    return False


# ============================================================
# ARTICLE EXTRACTION
# ============================================================

def extract_article_content(url: str):
    """
    Fetch and extract main article content from a URL.

    Returns tuple: (article_text, title, error_message)
      - article_text: str or None
      - title: str or None
      - error_message: str or None
    """
    try:
        if not url or not url.strip():
            return None, None, "No URL provided."

        url = url.strip()

        # Add protocol if missing
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # Normalize trivial fragments like trailing '#'
        parsed = urlparse(url)
        if parsed.fragment and parsed.fragment in ("", "/"):
            url = url.split("#", 1)[0]

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/119.0 Safari/537.36"
            )
        }

        # Stream to avoid downloading giant pages into memory
        response = requests.get(url, headers=headers, timeout=10, stream=True)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "") or ""
        if "html" not in content_type.lower():
            return None, None, "URL does not appear to contain readable HTML content."

        # Limit content to a safe maximum (e.g., 1 MB)
        max_bytes = 1_000_000  # 1 MB
        chunks = []
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                break
            downloaded += len(chunk)
            if downloaded > max_bytes:
                return None, None, "Page content too large to analyze safely."
            chunks.append(chunk)

        raw_bytes = b"".join(chunks)

        # Decode using response encoding if available, else utf-8 with fallback
        encoding = response.encoding or "utf-8"
        try:
            html = raw_bytes.decode(encoding, errors="ignore")
        except LookupError:
            html = raw_bytes.decode("utf-8", errors="ignore")

        # Parse HTML
        soup = BeautifulSoup(html, "html.parser")

        # Extract <title>
        title_tag = soup.find("title")
        title = title_tag.get_text().strip() if title_tag else "No Title Found"

        # Remove non-content elements
        for tag in soup(
            [
                "script",
                "style",
                "nav",
                "header",
                "footer",
                "aside",
                "form",
                "noscript",
            ]
        ):
            tag.decompose()

        # Try to locate the main article container
        article_content = None
        selectors = [
            "article",
            '[role="main"]',
            '[role="article"]',
            '[itemprop="articleBody"]',
            ".article-content",
            ".article__content",
            ".post-content",
            ".entry-content",
            ".story-body",
            ".article-body",
            ".content__article-body",
            "main",
        ]

        for selector in selectors:
            node = soup.select_one(selector)
            if node and node.find_all("p"):
                article_content = node
                break

        # Fallback: use body if nothing else looks like an article
        if not article_content:
            article_content = soup.find("body")

        if not article_content:
            return None, title, "Could not extract article content from the page."

        # Extract paragraphs
        paragraphs = article_content.find_all("p")
        texts = []
        for p in paragraphs:
            txt = p.get_text(separator=" ", strip=True)
            if txt:
                texts.append(txt)

        text = " ".join(texts).strip()
        text = re.sub(r"\s+", " ", text)

        if len(text) < 40:
            return None, title, "Article content too short or could not be extracted properly."

        # Optional: truncate extremely long text (your classifier only uses first ~1000 chars)
        max_chars = 15_000  # ~15k chars is plenty
        if len(text) > max_chars:
            text = text[:max_chars]

        return text, title, None

    except requests.exceptions.Timeout:
        return None, None, "⏱️ Request timed out. The website took too long to respond."
    except requests.exceptions.ConnectionError:
        return None, None, "🔌 Connection error. Could not reach the website."
    except requests.exceptions.HTTPError as e:
        return None, None, f"❌ HTTP Error {e.response.status_code}: Could not fetch the page."
    except Exception as e:
        return None, None, f"❌ Error fetching content: {str(e)}"


# ============================================================
# URL NORMALIZATION
# ============================================================

def normalize_url(text: str) -> str:
    """
    Clean and normalize URL-like input.
    Removes prefixes like 'url:', 'link:', 'source:' that users often type.
    """
    if not text:
        return ""

    text = text.strip()
    text = re.sub(r"^(url:|link:|source:)\s*", "", text, flags=re.IGNORECASE)
    return text


# ============================================================
# SIMPLE TESTS
# ============================================================

if __name__ == "__main__":
    # Test URL detection
    test_inputs = [
        "https://www.bbc.com/news/world",
        "bbc.com/news/article",
        "This is just plain news text about something",
        "www.cnn.com",
        "Check out this article at nytimes.com",
        "not a url at all",
    ]

    print("URL Detection Tests:\n")
    for inp in test_inputs:
        result = is_url(inp)
        print(f"'{inp[:60]}...' -> {'URL' if result else 'TEXT'}")

    print("\n" + "=" * 50 + "\n")

    # Test article extraction
    test_url = "https://www.bbc.com/news"
    print(f"Testing article extraction from: {test_url}\n")

    text, title, error = extract_article_content(test_url)

    if error:
        print(f"Error: {error}")
    else:
        print(f"Title: {title}")
        print(f"Content length: {len(text)} characters")
        print(f"Preview: {text[:200]}...")
