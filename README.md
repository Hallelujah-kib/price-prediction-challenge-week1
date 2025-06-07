![CI](https://github.com/Hallelujah-kib/price-prediction-challenge-week1/actions/workflows/ci.yml/badge.svg)
# 📈 Predicting Stock Price Movements Using News Sentiment

## 🔍 Overview

This project is part of **10 Academy's Week 1 Challenge**, which aims to predict short-term stock price movements by analyzing news headlines and computing technical indicators. The project integrates **Natural Language Processing (NLP)**, **Time Series Analysis**, and **Quantitative Finance** to explore the relationship between **news sentiment** and **stock price returns** for 7 companies.

---

## 🧠 Objectives

- Perform **exploratory data analysis** on stock prices and news articles.
- Compute **technical indicators** (SMA, EMA, RSI, MACD) using `TA-Lib`.
- Extract **financial metrics** using `PyNance`.
- Conduct **sentiment analysis** on financial news headlines.
- Align stock and news data by date and perform **correlation analysis**.

---

## 🛠️ Tech Stack

| Domain | Tools & Libraries |
|--------|--------------------|
| Language | Python 3.10+ |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Technical Indicators | [TA-Lib](https://mrjbq7.github.io/ta-lib/), [PyNance](https://pynance.readthedocs.io) |
| Sentiment Analysis | NLTK, TextBlob |
| DevOps | Git, GitHub, VS Code |
| Environment | Conda / venv |

---

## 📂 Project Structure

├── data/
│ ├── raw/
│ └── processed/
├── notebooks/
│ ├── eda_news_analysis.ipynb
│ ├── stock_indicators.ipynb
│ └── sentiment_correlation.ipynb
├── reports/
├── src/
│ ├── data_processor/
│ ├── indicators.py
│ ├── sentiment_correlation.py
│ └── utils.py
├── requirements.txt
├── .gitignore
└── README.md


> ✅ Code is modular, documented, and reusable across all 7 companies.

---

## 🧪 Notebooks Overview

### `1_eda_news_analysis.ipynb`
- Preprocess stock and news datasets
- Perform univariate and bivariate analysis
- Generate insights with visualizations

### `2_stock_indicators.ipynb`
- Load stock price data
- Compute SMA, EMA, RSI, MACD with TA-Lib
- Visualize each indicator
- Store processed output for correlation step

### `3_sentiment_correlation.ipynb`
- Analyze news sentiment using TextBlob
- Align and merge with stock returns
- Compute correlation coefficients
- Discuss implications of sentiment on stock performance

---

## 🧩 Reusability & Design

- 🔄 `StockIndicatorAnalyzer` class in `indicators.py` supports all companies with plug-and-play functionality.
- 🧱 Helper functions for loading/saving data are separated in `utils.py`.
- 🗃️ Clear separation of concerns across modules (`eda`, `sentiment`, `correlation`).

---

## 📈 Sample Plots

- 📊 **SMA/EMA Overlay**  
- 📉 **RSI Momentum Tracker**
- 🔃 **MACD Histogram**
- 📌 **Sentiment vs. Return Correlation Heatmap**

---

## ✅ How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Hallelujah-kib/price-prediction-challenge-week1.git
   cd price-prediction-challenge-week1

2. **Create and Activate a Virtual Environment**
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate

3. **Install Dependencies**
   pip install -r requirements.txt

4. **Run Notebooks**
   Open any of the notebooks in notebooks/ using Jupyter or VS Code and run each section interactively.

**💡 Key Learnings**
   - Used modular and object-oriented design for analytical workflows.
   - Applied real-world financial analysis using TA-Lib and PyNance.
   - Gained insight into how market sentiment impacts price fluctuations.
   - Documented and version-controlled the entire workflow using Git best practices

**🙌 Acknowledgements**
   Special thanks to the 10 Academy team and mentors for crafting such a well-structured challenge. Also thanks to the maintainers of open-source libraries used in this project.

**📃 License**
This project is licensed under the MIT License.
