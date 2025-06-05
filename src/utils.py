import pandas as pd
import os


def load_stock_data(ticker: str, filepath: str) -> pd.DataFrame:
    """
    Load stock price data from a CSV file.

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (for reference/logging).
    filepath : str
        Path to the CSV file containing stock price data.

    Returns:
    --------
    pd.DataFrame
        DataFrame sorted by date, containing stock price data.
    """
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def save_processed_data(df: pd.DataFrame, ticker: str, folder: str = "../data/processed") -> None:
    """
    Save a DataFrame to a CSV file in a specified folder.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save.
    ticker : str
        Stock ticker symbol used to name the saved file.
    folder : str, optional
        Folder path where the file will be saved (default is '../data/processed').

    Returns:
    --------
    None
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, f"{ticker}_processed.csv")
    df.to_csv(filepath, index=False)
