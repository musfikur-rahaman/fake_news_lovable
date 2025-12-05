# download_models.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def download_fake_news_model():
    model_name = "mrm8488/bert-tiny-finetuned-fake-news-detection"
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForSequenceClassification.from_pretrained(model_name)

def download_sentiment_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForSequenceClassification.from_pretrained(model_name)

if __name__ == "__main__":
    download_fake_news_model()
    download_sentiment_model()
    print("✅ Models downloaded into local HF cache")
