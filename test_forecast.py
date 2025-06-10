# test_forecast_models.py

from forecast_model import get_stock_data, run_garch_forecast, price_american_option_qmc

# Define the forecasting plan
plan = {
    "ticker": "AMZN",
    "start_date": "2024-01-01",
    "end_date": "2025-01-01",
    "forecast_horizon": 7
}

# Load stock data and calculate returns
data = get_stock_data(plan["ticker"], plan["start_date"], plan["end_date"])

# Run GARCH forecast
forecast, best_order, garch_params = run_garch_forecast(data, plan["forecast_horizon"], plan["ticker"])


print(f"\nBest GARCH Order: {best_order}")
print("Plots saved as 'garch_forecast_plot.png' and 'garch_diagnostics.png'")

# === American Option Pricing ===
S0 = float(data['Close'].iloc[-1])  # Convert to float
K = float(S0 * 1.05)  # 5% out-of-the-money strike, convert to float
r = 0.05       # Annual risk-free rate
T = 0.25       # 3 months to maturity
steps = 1024   # Number of simulation paths (power of 2)

# Calculate American option prices
result = price_american_option_qmc(S0, K, r, T, steps, 'call', garch_params)

# Print results with proper float conversion
print(f"\nEstimated American Call Option Price (K={K:.2f}, T={T} yr): ${float(result['option_price']):.4f}")
print(f"Standard Error: ${float(result['standard_error']):.4f}")

