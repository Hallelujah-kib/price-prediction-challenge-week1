import pandas as pd
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from typing import Tuple, List

def load_and_parse_data(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV and parses the 'date' column into datetime with EAT timezone.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Parsed DataFrame with a timezone-aware 'date' column.
    """
    df = pd.read_csv(file_path)

    try:
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', utc=True)
    except ValueError:
        df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
        if df['date'].isna().any():
            print("[Warning] Some dates could not be parsed and were set to NaT.")

    df['date'] = df['date'].dt.tz_convert('Etc/GMT+3')
    return df

def compute_descriptive_stats(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Computes headline length stats and top publisher counts.

    Args:
        df (pd.DataFrame): The input data.

    Returns:
        Tuple containing:
            - Descriptive stats of headline lengths
            - Top 5 publishers by frequency
    """
    df['headline_length'] = df['headline'].apply(len)
    return df['headline_length'].describe(), df['publisher'].value_counts().head(5)

def compute_sentiment_counts(df: pd.DataFrame) -> pd.Series:
    """
    Adds a sentiment column based on polarity score and returns sentiment frequency.

    Args:
        df (pd.DataFrame): The input data with 'headline'.

    Returns:
        pd.Series: Sentiment frequency counts.
    """
    def analyze_text(text: str) -> str:
        try:
            polarity = TextBlob(text).sentiment.polarity
            return 'positive' if polarity > 0 else 'negative' if polarity < 0 else 'neutral'
        except Exception:
            return 'neutral'

    df['sentiment'] = df['headline'].apply(analyze_text)
    return df['sentiment'].value_counts()

def extract_top_keywords(df: pd.DataFrame, top_n: int = 10) -> List[Tuple[str, int]]:
    """
    Extracts top N keywords from headlines after removing stopwords and punctuation.

    Args:
        df (pd.DataFrame): The input data.
        top_n (int): Number of top keywords to return.

    Returns:
        List of tuples: (keyword, frequency)
    """
    stop_words = set(stopwords.words('english'))
    words = []

    for headline in df['headline']:
        try:
            tokens = word_tokenize(str(headline).lower())
            words.extend([word for word in tokens if word.isalnum() and word not in stop_words])
        except Exception:
            continue

    return Counter(words).most_common(top_n)
