import pandas as pd
from textblob import TextBlob
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def load_news_data(filepath: str) -> pd.DataFrame:
    """
    Load news data CSV containing at least ['date', 'headline'] columns.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower()  # Convert all column names to lowercase
    return df


def preprocess_news_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare news data.
    """
    df = df.dropna(subset=["headline", "date"])
    return df


def compute_sentiment_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply TextBlob sentiment polarity scoring on the 'headline' column.
    Adds a 'sentiment_score' column with polarity scores between -1 and 1.
    """
    df["sentiment_score"] = df["headline"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return df


def aggregate_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment scores by date.
    Handles both datetime formats (with and without timezone offsets).
    """
    # Parse dates with flexible format handling
    news_df['date'] = pd.to_datetime(news_df['date'], format='mixed', utc=True)
    
    # Convert to naive datetime (remove timezone) and normalize to date
    news_df['date'] = news_df['date'].dt.tz_localize(None).dt.normalize()
    
    # Group by date and calculate mean sentiment
    daily_sentiment = news_df.groupby('date')['sentiment_score'].mean().reset_index()
    
    return daily_sentiment


def load_stock_returns(filepath: str) -> pd.DataFrame:
    """
    Load stock data CSV and compute daily returns.
    Expects columns containing 'date' and 'close'.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower()  # Convert all column names to lowercase
    
    # Parse dates and sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Calculate daily returns
    df['daily_return'] = df['close'].pct_change()
    
    return df[['date', 'daily_return']].dropna()


def align_data(stock_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge stock daily returns and daily sentiment by date.
    """
    # Ensure both date columns are datetime type
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    
    # Merge the dataframes
    merged = pd.merge(stock_df, sentiment_df, on="date", how="inner")
    return merged


def compute_correlation(merged_df: pd.DataFrame) -> float:
    """
    Compute Pearson correlation between daily returns and sentiment.
    """
    corr, _ = pearsonr(merged_df["daily_return"], merged_df["sentiment_score"])
    return corr


def plot_correlation(merged_df: pd.DataFrame, ticker: str):
    """
    Scatter plot of sentiment vs daily returns with correlation coefficient annotation.
    """
    corr = compute_correlation(merged_df)
    plt.figure(figsize=(8, 6))
    plt.scatter(merged_df["sentiment_score"], merged_df["daily_return"], alpha=0.6)
    plt.title(f"{ticker} - Correlation between News Sentiment and Daily Returns\nCorrelation = {corr:.3f}")
    plt.xlabel("Average Daily Sentiment")
    plt.ylabel("Daily Return")
    plt.grid(True)
    plt.show()