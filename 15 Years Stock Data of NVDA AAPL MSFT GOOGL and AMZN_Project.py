import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and Prepare Data
tickers = ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'NVDA']
close_cols = [f'Close_{ticker}' for ticker in tickers]

file_path = "C:/Users/nioan/OneDrive/Υπολογιστής/Projects/Python (15 Years Stock Data of NVDA AAPL MSFT GOOGL and AMZN)/15 Years Stock Data of NVDA AAPL MSFT GOOGL and AMZN.csv"
df = pd.read_csv(file_path)

# Convert 'Date' to datetime format and set it as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Check for missing values
print("Missing values:\n", df[close_cols].isnull().sum())

# ROI Analysis
def calculate_roi(df, tickers, initial_investment=1000):
    close_cols = [f'Close_{ticker}' for ticker in tickers]
    start_prices = df.iloc[0][close_cols]
    end_prices = df.iloc[-1][close_cols]
    shares_bought = initial_investment / start_prices
    final_values = shares_bought * end_prices
    roi = ((final_values - initial_investment) / initial_investment) * 100

    roi_df = pd.DataFrame({
        'Start_Price (€)': start_prices.round(2),
        'End_Price (€)': end_prices.round(2),
        'Shares_Bought': shares_bought.round(2),
        'Final_Value (€)': final_values.round(2),
        'ROI (%)': roi.round(2)
    })
    return roi_df

roi_df = calculate_roi(df, tickers)
print("\nROI Analysis:\n", roi_df)

# Year-over-Year Returns
def calculate_yoy_returns(df, close_cols):
    annual_close = df[close_cols].resample('Y').last()
    yoy_returns = annual_close.pct_change().dropna() * 100
    return yoy_returns.round(2)

yoy_returns = calculate_yoy_returns(df, close_cols)
print("\nYoY Returns:\n", yoy_returns)

# Plot YoY
plt.figure(figsize=(12, 6))
for col in yoy_returns.columns:
    plt.plot(yoy_returns.index.year, yoy_returns[col], marker='o', label=col.replace("Close_", ""))
plt.title("Year-over-Year Returns (%) by Stock")
plt.xlabel("Year")
plt.ylabel("Return (%)")
plt.legend(title="Stock")
plt.grid(True)
plt.tight_layout()
plt.show()

# Drawdown Analysis
def calculate_drawdowns(df, tickers):
    drawdowns = pd.DataFrame(index=df.index)
    for ticker in tickers:
        col = f'Close_{ticker}'
        running_max = df[col].cummax()
        drawdown = (df[col] - running_max) / running_max * 100
        drawdowns[ticker] = drawdown
    return drawdowns

drawdowns = calculate_drawdowns(df, tickers)
max_drawdowns = drawdowns.min().round(2)
drawdown_dates = drawdowns.idxmin()
drawdown_summary = pd.DataFrame({
    'Max Drawdown (%)': max_drawdowns,
    'Date of Max Drawdown': drawdown_dates
})

print("\nDrawdown Summary:\n", drawdown_summary)

plt.figure(figsize=(10, 6))
bars = plt.bar(drawdown_summary.index, drawdown_summary["Max Drawdown (%)"], color='tomato')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.1f}%', ha='center', va='bottom' if yval > -50 else 'top')
plt.title("Maximum Drawdown per Stock (2010–2025)")
plt.ylabel("Drawdown (%)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Seasonality Heatmap
monthly_returns = df[close_cols].copy()
monthly_returns['Month'] = monthly_returns.index.month
daily_returns = monthly_returns.drop(columns='Month').pct_change().dropna()
daily_returns['Month'] = monthly_returns['Month'].iloc[1:]

avg_monthly_returns = (daily_returns.groupby('Month').mean() * 100).round(2).T
print("\nAverage Monthly Returns (Heatmap Data):\n", avg_monthly_returns)

plt.figure(figsize=(12, 5))
sns.heatmap(avg_monthly_returns, annot=True, fmt=".2f", cmap="RdYlGn", linewidths=0.5, cbar_kws={'label': 'Avg Daily Return (%)'})
plt.title("Seasonality Heatmap: Average Daily Returns by Month")
plt.xlabel("Month")
plt.ylabel("Stock")
plt.tight_layout()
plt.show()
