import pandas as pd
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

def analyze_data(file_path):
    df = pd.read_csv(file_path)
    try:
        # Try parsing with the expected format
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', utc=True)
    except ValueError:
        # If parsing fails, fall back to inferring the format
        df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
        if df['date'].isna().any():
            print("Warning: Some dates could not be parsed and were set to NaT.")
    # Convert to EAT
    df['date'] = df['date'].dt.tz_convert('Etc/GMT+3')
    return df

def get_descriptive_stats(df):
    df['headline_length'] = df['headline'].apply(len)
    return df['headline_length'].describe(), df['publisher'].value_counts().head(5)

def get_sentiment(df):
    def analyze_text(text):
        try:
            polarity = TextBlob(text).sentiment.polarity
            return 'positive' if polarity > 0 else 'negative' if polarity < 0 else 'neutral'
        except:
            return 'neutral'
    df['sentiment'] = df['headline'].apply(analyze_text)
    return df['sentiment'].value_counts()

def get_keywords(df):
    stop_words = set(stopwords.words('english'))
    words = []
    for headline in df['headline']:
        tokens = word_tokenize(headline.lower())
        words.extend([word for word in tokens if word.isalnum() and word not in stop_words])
    return Counter(words).most_common(10)