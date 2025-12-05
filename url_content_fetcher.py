import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse

def is_url(text):
    """
    Detect if the input text is a URL.
    """
    # Check for common URL patterns
    url_pattern = re.compile(
        r'^(https?://)?(www\.)?'  # http:// or https:// or www.
        r'[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)+'  # domain name
        r'(/.*)?$',  # optional path
        re.IGNORECASE
    )
    
    # Also check if it contains common TLDs
    tld_pattern = re.compile(r'\.(com|org|net|edu|gov|uk|co|io|ai|news|info|biz)', re.IGNORECASE)
    
    text = text.strip()
    return bool(url_pattern.match(text)) or bool(tld_pattern.search(text))

def extract_article_content(url):
    """
    Fetch and extract main article content from a URL.
    Returns tuple: (article_text, title, error_message)
    """
    try:
        # Add protocol if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Set headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch the page with timeout
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title = soup.find('title')
        title = title.get_text().strip() if title else "No Title Found"
        
        # Remove script and style elements
        for script in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            script.decompose()
        
        # Try to find article content using common tags
        article_content = None
        
        # Priority order of selectors
        selectors = [
            'article',
            '[role="main"]',
            '.article-content',
            '.post-content',
            '.entry-content',
            '.story-body',
            '.article-body',
            'main',
        ]
        
        for selector in selectors:
            article_content = soup.select_one(selector)
            if article_content:
                break
        
        # If no article found, try to find paragraphs
        if not article_content:
            article_content = soup.find('body')
        
        if not article_content:
            return None, title, "Could not extract article content from the page."
        
        # Extract paragraphs
        paragraphs = article_content.find_all('p')
        text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Check if we got meaningful content
        if len(text) < 10:
            return None, title, "Article content too short or could not be extracted properly."
        
        return text, title, None
        
    except requests.exceptions.Timeout:
        return None, None, "⏱️ Request timed out. The website took too long to respond."
    except requests.exceptions.ConnectionError:
        return None, None, "🔌 Connection error. Could not reach the website."
    except requests.exceptions.HTTPError as e:
        return None, None, f"❌ HTTP Error {e.response.status_code}: Could not fetch the page."
    except Exception as e:
        return None, None, f"❌ Error fetching content: {str(e)}"

def normalize_url(text):
    """
    Clean and normalize URL input.
    """
    text = text.strip()
    
    # Remove common prefixes people might type
    text = re.sub(r'^(url:|link:|source:)\s*', '', text, flags=re.IGNORECASE)
    
    return text

# Test function
if __name__ == "__main__":
    # Test URL detection
    test_inputs = [
        "https://www.bbc.com/news/world",
        "bbc.com/news/article",
        "This is just plain news text about something",
        "www.cnn.com",
        "Check out this article at nytimes.com"
    ]
    
    print("URL Detection Tests:\n")
    for inp in test_inputs:
        result = is_url(inp)
        print(f"'{inp[:50]}...' -> {'URL' if result else 'TEXT'}")
    
    print("\n" + "="*50 + "\n")
    
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