import pandas as pd
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from typing import Dict, List, Tuple
from dateutil import tz

class DataAnalyzer:
    """
    A class to perform exploratory data analysis and sentiment analysis
    on financial news headline data.

    Attributes:
        df (pd.DataFrame): The loaded and processed DataFrame.
    """

    def __init__(self, file_path: str):
        """
        Initializes the DataAnalyzer with data loaded from a CSV file.
        Parses the 'date' column into timezone-aware datetimes and converts to EAT.
        """
        try:
            self.df = pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"[Error] File not found at: {file_path}")
        except Exception as e:
            raise Exception(f"[Error] Failed to read CSV: {str(e)}")

        try:
            # Parse all date strings to datetime (naive and timezone-aware)
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce', utc=True)

            # Handle any unparseable rows
            num_invalid = self.df['date'].isna().sum()
            if num_invalid > 0:
                print(f"[Warning] Dropping {num_invalid} rows with invalid dates.")
                self.df.dropna(subset=['date'], inplace=True)

                # Localize naive timestamps to assumed UTC-4 (Eastern Time in dataset)
                mask_naive = self.df['date'].dt.tz is None
                # self.df.loc[mask_naive, 'date'] = self.df.loc[mask_naive, 'date'].dt.tz_localize('America/New_York')

                # # Convert all timestamps to EAT (East Africa Time)
                # self.df['date'] = self.df['date'].dt.tz_convert('Africa/Nairobi')

            print(f"PRINTIG: NUM OF INVALD { num_invalid}")

        except Exception as e:
            raise Exception(f"[Error] Failed to parse date column: {str(e)}")

        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

    def get_descriptive_stats(self) -> Dict[str, pd.Series]:
        """
        Computes descriptive statistics about headline lengths and publisher frequency.

        Returns:
            dict: Contains descriptive stats for headline lengths and top 5 publisher counts.
        """
        self.df['headline_length'] = self.df['headline'].apply(len)
        return {
            'headline_length': self.df['headline_length'].describe(),
            'publisher_counts': self.df['publisher'].value_counts().head(5)
        }

    def get_sentiment(self) -> pd.Series:
        """
        Analyzes sentiment polarity of headlines using TextBlob and classifies them.

        Returns:
            pd.Series: Frequency count of sentiment labels (positive, negative, neutral).
        """
        def analyze_sentiment(text: str) -> float:
            """
            Helper function to extract sentiment polarity from text.

            Args:
                text (str): Headline text.

            Returns:
                float: Sentiment polarity score between -1 and 1.
            """
            try:
                return TextBlob(text).sentiment.polarity
            except Exception:
                return 0.0  # fallback to neutral if analysis fails

        self.df['sentiment_score'] = self.df['headline'].apply(analyze_sentiment)
        self.df['sentiment'] = self.df['sentiment_score'].apply(
            lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral'
        )
        return self.df['sentiment'].value_counts()

    def get_keywords(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Extracts top N most common keywords from the headline text.

        Args:
            top_n (int): Number of top keywords to return.

        Returns:
            list: A list of tuples with keyword and their frequency count.
        """
        stop_words = set(stopwords.words('english'))
        words = []

        for headline in self.df['headline']:
            try:
                tokens = word_tokenize(str(headline).lower())
                filtered = [word for word in tokens if word.isalnum() and word not in stop_words]
                words.extend(filtered)
            except Exception:
                continue  # skip malformed headline rows

        return Counter(words).most_common(top_n)

    def get_dataframe(self) -> pd.DataFrame:
        """
        Returns the processed internal DataFrame.

        Returns:
            pd.DataFrame: The processed data.
        """
        return self.df


# Usage Example
if __name__ == "__main__":
    file_path = 'data/raw_analyst_ratings.csv'
    analyzer = DataAnalyzer(file_path)

    stats = analyzer.get_descriptive_stats()
    print("Headline Length Stats:\n", stats['headline_length'])
    print("\nTop Publishers:\n", stats['publisher_counts'])

    sentiment = analyzer.get_sentiment()
    print("\nSentiment Counts:\n", sentiment)

    keywords = analyzer.get_keywords()
    print("\nTop Keywords:\n", keywords)
