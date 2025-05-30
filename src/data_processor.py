import pandas as pd
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

class DataAnalyzer:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.df['date'] = pd.to_datetime(self.df['date'], utc=True).dt.tz_convert('Etc/GMT+3')
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

    def get_descriptive_stats(self):
        self.df['headline_length'] = self.df['headline'].apply(len)
        return {
            'headline_length': self.df['headline_length'].describe(),
            'publisher_counts': self.df['publisher'].value_counts().head(5)
        }

    def get_sentiment(self):
        def analyze_sentiment(text):
            try:
                polarity = TextBlob(text).sentiment.polarity
                return 'positive' if polarity > 0 else 'negative' if polarity < 0 else 'neutral'
            except:
                return 'neutral'
        self.df['sentiment'] = self.df['headline'].apply(analyze_sentiment)
        return self.df['sentiment'].value_counts()

    def get_keywords(self):
        stop_words = set(stopwords.words('english'))
        words = []
        for headline in self.df['headline']:
            tokens = word_tokenize(headline.lower())
            words.extend([word for word in tokens if word.isalnum() and word not in stop_words])
        return Counter(words).most_common(10)

# Usage example
if __name__ == "__main__":
    analyzer = DataAnalyzer('data/raw_analyst_ratings.csv')
    stats = analyzer.get_descriptive_stats()
    sentiment = analyzer.get_sentiment()
    keywords = analyzer.get_keywords()
    print(stats, sentiment, keywords)