import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download historical data for AAPL
df = yf.download("AAPL", start="2021-01-01", end="2023-01-01")

# Calculate the short-term and long-term moving averages
df['Short_MA'] = df['Close'].rolling(window=20).mean()
df['Long_MA'] = df['Close'].rolling(window=50).mean()

# Drop rows with NaN values from moving averages
df.dropna(inplace=True)

# Create signals: 1 when Short_MA > Long_MA, else 0
df['Signal'] = (df['Short_MA'] > df['Long_MA']).astype(int)

# Find where signals change (crossovers)
df['Position'] = df['Signal'].diff()

# Print buy/sell signals for debugging
print("Buy/Sell signals:")
print(df[df['Position'] != 0][['Close', 'Short_MA', 'Long_MA', 'Signal', 'Position']])

# === BACKTESTING ===

# 1. Calculate daily market returns
df['Market_Return'] = df['Close'].pct_change()

# 2. Calculate strategy returns
# When Signal == 1, you hold stock, so earn the market return
# Shift signal by 1 to simulate buying at close previous day
df['Strategy_Return'] = df['Market_Return'] * df['Signal'].shift(1)

# 3. Calculate cumulative returns
df['Market_Cum_Returns'] = (1 + df['Market_Return']).cumprod() - 1
df['Strategy_Cum_Returns'] = (1 + df['Strategy_Return']).cumprod() - 1

# 4. Print final returns
print(f"\nMarket Return: {df['Market_Cum_Returns'].iloc[-1]*100:.2f}%")
print(f"Strategy Return: {df['Strategy_Cum_Returns'].iloc[-1]*100:.2f}%")

# === PLOTTING ===

plt.figure(figsize=(14,7))
plt.plot(df['Close'], label='Close Price', alpha=0.5)
plt.plot(df['Short_MA'], label='20-day MA')
plt.plot(df['Long_MA'], label='50-day MA')

# Plot buy signals (Position == 1)
plt.plot(df[df['Position'] == 1].index, 
         df['Short_MA'][df['Position'] == 1], 
         '^', markersize=12, color='g', label='Buy Signal')

# Plot sell signals (Position == -1)
plt.plot(df[df['Position'] == -1].index, 
         df['Short_MA'][df['Position'] == -1], 
         'v', markersize=12, color='r', label='Sell Signal')

plt.title("AAPL Moving Average Crossover Strategy")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.show()