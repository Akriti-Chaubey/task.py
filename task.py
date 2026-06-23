# ==========================================================
# BITCOIN MARKET SENTIMENT VS TRADER PERFORMANCE ANALYSIS
# ==========================================================

# Install packages if needed:
# pip install pandas numpy matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
print ("Current Folder:")
print(os.getcwd())

print("\nFiles Available:")
print(os.listdir())


# ==========================================================
# STEP 1: ADD YOUR DATASET LINKS HERE
# ==========================================================

SENTIMENT_DATASET_LINK = "data.csv"

TRADER_DATASET_LINK = "historical_data.csv"

# ==========================================================
# STEP 2: LOAD DATASETS
# ==========================================================

print("Loading datasets...")

sentiment = pd.read_csv(SENTIMENT_DATASET_LINK)
trader = pd.read_csv(TRADER_DATASET_LINK, on_bad_lines='skip')

print("\nSentiment Dataset Shape:", sentiment.shape)
print("Trader Dataset Shape:", trader.shape)

# ==========================================================
# STEP 3: DISPLAY DATA
# ==========================================================

print("\nSentiment Dataset")
print(sentiment.head())

print("\nTrader Dataset")
print(trader.head())

# ==========================================================
# STEP 4: DATA CLEANING
# ==========================================================

print("\nCleaning data...")

sentiment.columns = sentiment.columns.str.strip()
trader.columns = trader.columns.str.strip()


sentiment['date'] = pd.to_datetime(sentiment['date']).dt.strftime('%Y-%m-%d')
trader['Timestamp'] = pd.to_datetime(trader['Timestamp'], unit='ms', errors='coerce')
trader['Date'] = trader['Timestamp'].dt.strftime('%Y-%m-%d')

print("Cleaning Completed")

# =======================================================
# STEP 5: MERGE DATASETS
# =======================================================
print("\nMerging datasets...")


merged = pd.merge(sentiment, trader, left_on='date', right_on='Date')
print(f"Merged Dataset Rows: {len(merged)}")

# =======================================================
# STEP 6: PLOT DATA
# =======================================================
print("\nGenerating Chart...")


merged.columns = merged.columns.str.lower()


plt.figure(figsize=(10, 6))


sns.countplot(data=merged, x='classification')

plt.title("Market Sentiment Distribution")
plt.xlabel("classification")
plt.ylabel("Count")

plt.show()

# ==========================================================
# STEP 8: PROFITABILITY ANALYSIS
# ==========================================================

print("\nProfitability Analysis")


merged.columns = merged.columns.str.lower().str.strip()


profit_analysis = merged.groupby('classification')['closed pnl'].mean()
print(profit_analysis)

# ==========================================================
# STEP 9: WIN RATE ANALYSIS
# ==========================================================

merged['Win'] = merged['closed pnl'] > 0

win_rate = merged.groupby(
    'classification'
)['Win'].mean() * 100

print("\nWin Rate (%)")
print(win_rate)

plt.figure(figsize=(8,5))
sns.barplot(
    x=win_rate.index,
    y=win_rate.values
)

plt.title("Win Rate by Sentiment")
plt.ylabel("Win Rate (%)")
plt.show()

# ==========================================================
# STEP 10: LEVERAGE ANALYSIS
# ==========================================================

if 'leverage' in merged.columns:

    leverage_stats = merged.groupby(
        'classification'
    )['leverage'].mean()

    print("\nAverage Leverage")
    print(leverage_stats)

    plt.figure(figsize=(8,5))
    sns.boxplot(
        x='classification',
        y='leverage',
        data=merged
    )

    plt.title("Leverage Distribution")
    plt.show()

# ==========================================================
# STEP 11: TRADING VOLUME ANALYSIS
# ==========================================================

if 'size' in merged.columns:

    volume = merged.groupby(
        'classification'
    )['size'].sum()

    print("\nTrading Volume")
    print(volume)

    plt.figure(figsize=(8,5))
    sns.barplot(
        x=volume.index,
        y=volume.values
    )

    plt.title("Trading Volume by Sentiment")
    plt.ylabel("Volume")
    plt.show()

# ==========================================================
# STEP 12: TOP TRADERS
# ==========================================================

print("\nTop Traders Analysis")

top_traders = merged.groupby(
    'account'
)['closed pnl'].sum().sort_values(
    ascending=False
).head(20)

print(top_traders)

plt.figure(figsize=(12,6))
top_traders.plot(kind='bar')
plt.title("Top 20 Traders by Total pnl")
plt.ylabel("pnl")
plt.show()

# ==========================================================
# STEP 13: RISK ANALYSIS
# ==========================================================

if ('leverage' in merged.columns) and ('size' in merged.columns):

    merged['RiskScore'] = (
        merged['leverage'] *
        merged['size']
    )

    plt.figure(figsize=(10,6))

    sns.scatterplot(
        x='RiskScore',
        y='closed pnl',
        data=merged
    )

    plt.title("Risk Score vs Profit/Loss")
    plt.show()

# ==========================================================
# STEP 14: SYMBOL ANALYSIS
# ==========================================================

if 'symbol' in merged.columns:

    symbol_analysis = merged.groupby(
        ['symbol', 'classification']
    )['closed pnl'].mean().reset_index()

    print("\nSymbol Analysis")
    print(symbol_analysis.head())

# ==========================================================
# STEP 15: CORRELATION HEATMAP
# ==========================================================

numeric_columns = merged.select_dtypes(
    include=np.number
)

plt.figure(figsize=(10,8))

sns.heatmap(
    numeric_columns.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.show()

# ==========================================================
# STEP 16: OVERALL PERFORMANCE SUMMARY
# ==========================================================

print("\n=================================================")
print("OVERALL SUMMARY")
print("=================================================")

print("Total Trades:",
      len(merged))

print("Total PnL:",
      merged['closed pnl'].sum())

print("Average PnL:",
      merged['closed pnl'].mean())

print("Median PnL:",
      merged['closed pnl'].median())

print("Maximum Profit:",
      merged['closed pnl'].max())

print("Maximum Loss:",
      merged['closed pnl'].min())

print("Overall Win Rate:",
      round(
          merged['Win'].mean()*100,
          2
      ),
      "%"
)

# ==========================================================
# STEP 17: SAVE FINAL DATASET
# ==========================================================

merged.to_csv(
    "Merged_Trading_Sentiment_Data.csv",
    index=False
)

print("\nMerged dataset saved successfully!")
print("File Name: Merged_Trading_Sentiment_Data.csv")

# ==========================================================
# END OF PROJECT
# ==========================================================