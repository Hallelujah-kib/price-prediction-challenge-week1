import pandas as pd
import talib
import matplotlib.pyplot as plt


class StockIndicatorAnalyzer:
    """
    A class to calculate technical indicators for stock price data using TA-Lib.

    Attributes
    ----------
    df : pd.DataFrame
        DataFrame containing stock price data with columns ['Open', 'High', 'Low', 'Close', 'Volume'].

    Methods
    -------
    compute_indicators():
        Calculates SMA, EMA, RSI, MACD, and Signal line indicators and returns an augmented DataFrame.
    
    add_daily_return(df):
        Calculates the daily percentage return based on closing prices and returns the DataFrame with a new 'Daily_Return' column.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the StockIndicatorAnalyzer with stock price DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing stock price data. Must include 'Open', 'High', 'Low', 'Close', 'Volume' columns.
        """
        self.df = df.copy()

    def compute_indicators(self) -> pd.DataFrame:
        """
        Compute technical indicators using TA-Lib.

        Returns
        -------
        pd.DataFrame
            DataFrame with new columns: 'SMA_20', 'EMA_20', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist'.
        """
        self.df['SMA_20'] = talib.SMA(self.df['Close'], timeperiod=20)
        self.df['EMA_20'] = talib.EMA(self.df['Close'], timeperiod=20)
        self.df['RSI_14'] = talib.RSI(self.df['Close'], timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(self.df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        self.df['MACD'] = macd
        self.df['MACD_signal'] = macd_signal
        self.df['MACD_hist'] = macd_hist
        return self.df

    
    def add_daily_return(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the daily percentage return of the closing price.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing stock price data with a 'Close' column.

        Returns
        -------
        pd.DataFrame
            DataFrame with a new 'Daily_Return' column representing daily percentage changes.
        """
        df = df.copy()
        df['Daily_Return'] = df['Close'].pct_change() * 100
        return df


    def plot_sma_ema(self, df: pd.DataFrame, ticker: str) -> None:
        """
        Plot the closing price along with 20-day SMA and EMA.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing 'Close', 'SMA_20', and 'EMA_20' columns.
        ticker : str
            Stock ticker symbol for the plot title.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['Close'], label='Close Price')
        plt.plot(df['Date'], df['SMA_20'], label='SMA 20')
        plt.plot(df['Date'], df['EMA_20'], label='EMA 20')
        plt.title(f"{ticker} Close Price with SMA & EMA (20-day)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()


    def plot_rsi(self, df: pd.DataFrame, ticker: str) -> None:
        """
        Plot the Relative Strength Index (RSI).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing 'RSI_14' column.
        ticker : str
            Stock ticker symbol for the plot title.
        """
        plt.figure(figsize=(12, 4))
        plt.plot(df['Date'], df['RSI_14'], label='RSI 14')
        plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
        plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
        plt.title(f"{ticker} Relative Strength Index (RSI)")
        plt.xlabel("Date")
        plt.ylabel("RSI")
        plt.legend()
        plt.grid(True)
        plt.show()


    def plot_macd(self, df: pd.DataFrame, ticker: str) -> None:
        """
        Plot the MACD line, signal line, and histogram.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing 'MACD', 'MACD_signal', and 'MACD_hist' columns.
        ticker : str
            Stock ticker symbol for the plot title.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['MACD'], label='MACD')
        plt.plot(df['Date'], df['MACD_signal'], label='Signal Line')
        plt.bar(df['Date'], df['MACD_hist'], label='MACD Histogram', color='gray', alpha=0.5)
        plt.title(f"{ticker} MACD Indicator")
        plt.xlabel("Date")
        plt.ylabel("MACD")
        plt.legend()
        plt.grid(True)
        plt.show()
