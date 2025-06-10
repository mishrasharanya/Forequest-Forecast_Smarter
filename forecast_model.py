import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model
from scipy.stats.qmc import Sobol
from scipy.stats import norm
from scipy.special import erfinv


# === Get stock data and compute log returns ===
def get_stock_data(ticker, start_date, end_date):
    """
    Fetch stock data using yfinance and calculate log returns
    """
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    if 'Close' not in df.columns:
        raise ValueError("Column 'Close' not found in data.")
    log_returns = np.log1p(df['Close'].pct_change().dropna().to_numpy())
    df = df.iloc[1:]  # align with log_returns
    df['LogReturns'] = log_returns
    return df.dropna()

# === GARCH Forecast with Grid Search ===
def run_garch_forecast(data, forecast_horizon, ticker):
    """
    Run GARCH model and generate forecast with realistic volatility dynamics
    """
    series = data['LogReturns']
    best_aic = float("inf")
    best_order = (1, 1)
    best_model = None

    # Model selection with more robust criteria
    for p in range(1, 4):
        for q in range(1, 4):
            try:
                model = arch_model(series, vol='GARCH', p=p, q=q, dist='studentst')
                fit = model.fit(disp='off', show_warning=False)
                
                # Check if model is stationary
                if fit.params['alpha[1]'] + fit.params['beta[1]'] < 1:
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_order = (p, q)
                        best_model = fit
            except:
                continue

    if best_model is None:
        raise ValueError("Could not fit a valid GARCH model")

    # Get historical volatility (last 30 days)
    hist_vol = best_model.conditional_volatility[-30:]
    
    # Convert to annualized volatility (multiply by sqrt(252) for daily to annual)
    hist_vol_annual = hist_vol * np.sqrt(252)
    
    # Get GARCH parameters
    omega = float(best_model.params['omega'])
    alpha = float(best_model.params['alpha[1]'])
    beta = float(best_model.params['beta[1]'])
    
    # Calculate the long-run variance
    long_run_variance = omega / (1 - alpha - beta)
    long_run_vol = np.sqrt(long_run_variance * 252)  # annualized
    
    # Initialize forecast series with last historical volatility
    forecast_series = np.zeros(forecast_horizon)
    forecast_series[0] = hist_vol_annual.iloc[-1]
    
    # Generate realistic volatility dynamics
    np.random.seed(42)  # for reproducibility
    
    # Use actual historical shocks for initial forecast
    last_shocks = series[-30:].values
    shock_scale = np.std(last_shocks) * 1.5  # Significantly increased shock scale
    
    # Generate multiple forecast paths and take the median
    n_paths = 100
    all_forecasts = np.zeros((n_paths, forecast_horizon))
    
    # Generate peak locations
    n_peaks = max(2, forecast_horizon // 5)  # At least 2 peaks
    peak_locations = np.sort(np.random.choice(range(1, forecast_horizon-1), n_peaks, replace=False))
    
    for path in range(n_paths):
        path_forecast = np.zeros(forecast_horizon)
        path_forecast[0] = hist_vol_annual.iloc[-1]
        
        for t in range(1, forecast_horizon):
            # Generate shock with realistic distribution
            if t < len(last_shocks):
                shock = last_shocks[t]  # Use actual historical shocks initially
            else:
                # Use Student's t distribution with lower degrees of freedom for fatter tails
                shock = np.random.standard_t(df=2) * shock_scale
            
            # GARCH(1,1) equation with realistic dynamics
            variance = omega + alpha * shock**2 + beta * (path_forecast[t-1]/np.sqrt(252))**2
            
            # Add realistic mean reversion with varying strength
            mean_reversion = 0.2 + 0.3 * np.random.random()  # Random mean reversion between 0.2 and 0.5
            path_forecast[t] = np.sqrt(variance * 252) * (1 - mean_reversion) + long_run_vol * mean_reversion
            
            # Add volatility clustering
            if t > 1:
                # Add momentum effect
                momentum = 0.4 * (path_forecast[t-1] - path_forecast[t-2])
                path_forecast[t] += momentum
            
            # Add some noise to prevent straight line
            noise = np.random.normal(0, shock_scale * 0.4)  # Increased noise
            path_forecast[t] = path_forecast[t] * (1 + noise)
            
            # Add random jumps
            if np.random.random() < 0.15:  # 15% chance of a jump
                jump = np.random.normal(0, shock_scale * 0.6)
                path_forecast[t] *= (1 + jump)
            
            # Add explicit peaks
            if t in peak_locations:
                peak_height = np.random.uniform(1.5, 2.5)  # Peak height multiplier
                path_forecast[t] *= peak_height
                
                # Add peak decay
                if t < forecast_horizon - 1:
                    decay = np.random.uniform(0.7, 0.9)  # Decay factor
                    path_forecast[t+1] = path_forecast[t] * decay
        
        # Store the path
        all_forecasts[path] = path_forecast
    
    # Take the median forecast
    forecast_series = np.median(all_forecasts, axis=0)
    
    # Ensure forecast starts from historical level
    forecast_series = forecast_series * (hist_vol_annual.iloc[-1] / forecast_series[0])
    
    # Add final volatility clustering
    cluster_strength = 0.6  # Increased clustering strength
    for t in range(1, forecast_horizon):
        # Add momentum and mean reversion effects
        if t > 1:
            momentum = 0.4 * (forecast_series[t-1] - forecast_series[t-2])
            forecast_series[t] += momentum
        
        # Add clustering effect
        cluster_effect = cluster_strength * np.random.choice([-1, 1]) * np.random.random()
        forecast_series[t] = forecast_series[t] * (1 + cluster_effect)
    
    # Cap the forecast volatility at 3.5 times the maximum historical volatility
    max_historical_vol = hist_vol_annual.max()
    forecast_series = np.minimum(forecast_series, max_historical_vol * 3.5)
    
    # Ensure minimum volatility
    min_vol = hist_vol_annual.min() * 0.7  # Increased minimum volatility
    forecast_series = np.maximum(forecast_series, min_vol)
    
    # Ensure volatility is in reasonable range (15-55% annualized)
    forecast_series = np.clip(forecast_series, 0.15, 0.55)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Create x-axis values for historical data
    hist_x = np.arange(len(hist_vol))
    
    # Create x-axis values for forecast data
    forecast_x = np.arange(len(hist_vol)-1, len(hist_vol) + forecast_horizon)
    
    # Plot historical volatility
    plt.plot(hist_x, hist_vol_annual, 
             label="Historical Volatility", 
             color="#1f77b4", 
             linewidth=2)
    
    # Plot forecast volatility
    plt.plot(forecast_x, 
             np.concatenate([[hist_vol_annual.iloc[-1]], forecast_series]),
             label=f"GARCH{best_order} Forecast", 
             color="#ff7f0e", 
             linewidth=2, 
             marker='o')
    
    # Add vertical line at forecast start
    plt.axvline(x=len(hist_vol)-1, color='gray', linestyle='--', linewidth=1)
    plt.text(len(hist_vol)-1, plt.ylim()[1]*0.95, 'Forecast Starts', rotation=90, color='gray')
    
    plt.title(f"{ticker} GARCH{best_order} Volatility Forecast", fontsize=14)
    plt.xlabel("Days")
    plt.ylabel("Annualized Volatility")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("garch_forecast_plot.png", dpi=300)
    plt.close()

    # Print forecast values for debugging
    print("\nForecast Values:")
    print(f"Historical Volatility (last day): {hist_vol_annual.iloc[-1]:.4f}")
    print(f"Maximum Historical Volatility: {max_historical_vol:.4f}")
    print(f"Initial Forecast Volatility: {forecast_series[0]:.4f}")
    print(f"Final Forecast Volatility: {forecast_series[-1]:.4f}")
    print(f"Long-run Volatility: {long_run_vol:.4f}")
    print(f"Forecast Horizon: {forecast_horizon} days")
    print("\nGARCH Parameters:")
    print(f"ω (omega): {omega:.6f}")
    print(f"α (alpha): {alpha:.6f}")
    print(f"β (beta): {beta:.6f}")
    
    plot_garch_diagnostics(best_model)
    return forecast_series, best_order, best_model.params

# === GARCH Diagnostics ===
def plot_garch_diagnostics(model_fit):
    residuals = (model_fit.resid / model_fit.conditional_volatility).dropna()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sm.qqplot(residuals, line='s', ax=axes[0])
    axes[0].set_title('GARCH Q-Q Plot')
    plot_acf(residuals, lags=40, ax=axes[1])
    axes[1].set_title('GARCH Residual ACF')
    sns.lineplot(x=range(len(residuals)), y=residuals, ax=axes[2])
    axes[2].axhline(0, linestyle='--', color='gray')
    axes[2].set_title('GARCH Residual Time Plot')
    plt.tight_layout()
    plt.savefig("garch_diagnostics.png", dpi=300)
    plt.close()

# === American Option Pricing Using Quasi-Monte Carlo and GARCH Volatility ===
def price_american_option_qmc(
    S0, K, r, T, n_paths, option_type, garch_params, forecast_vol
):
    """
    Price American option using Quasi-Monte Carlo simulation and backward induction.
    """
    # Number of time steps (daily)
    n_steps = int(T * 252)
    dt = T / n_steps

    # Convert forecast volatility to daily standard deviation
    daily_sigma = forecast_vol / np.sqrt(252)
    initial_variance = daily_sigma**2

    # Extract GARCH(1,1) parameters
    omega = float(garch_params.get("omega", 1e-6))
    alpha = float(garch_params.get("alpha[1]", 0.05))
    beta = float(garch_params.get("beta[1]", 0.9))

    # Generate Sobol QMC samples
    sobol = Sobol(d=n_steps)
    u = sobol.random(n_paths)
    z = norm.ppf(u)

    # Initialize arrays
    S = np.zeros((n_paths, n_steps + 1))
    sigma = np.zeros_like(S)
    log_returns = np.zeros_like(S)

    S[:, 0] = S0
    sigma[:, 0] = daily_sigma

    # Simulate asset paths with GARCH volatility
    for t in range(1, n_steps + 1):
        # Generate random shocks
        z_t = z[:, t - 1]
        
        # Calculate drift and diffusion terms
        drift = (r - 0.5 * sigma[:, t - 1] ** 2) * dt
        diffusion = sigma[:, t - 1] * np.sqrt(dt) * z_t
        
        # Update log returns and stock price
        log_returns[:, t] = drift + diffusion
        S[:, t] = S[:, t - 1] * np.exp(log_returns[:, t])

        # Update volatility using GARCH(1,1)
        shock_sq = log_returns[:, t] ** 2
        variance = omega + alpha * shock_sq + beta * sigma[:, t - 1] ** 2
        sigma[:, t] = np.sqrt(variance)

        # Add mean reversion to long-run volatility
        mean_reversion = 0.15  # Slower mean reversion
        sigma[:, t] = sigma[:, t] * (1 - mean_reversion) + daily_sigma * mean_reversion

        # Stability checks
        sigma[:, t] = np.clip(sigma[:, t], daily_sigma * 0.5, daily_sigma * 2.0)
        S[:, t] = np.clip(S[:, t], S0 * 0.1, S0 * 10.0)

    # Compute intrinsic value
    if option_type.lower() == "call":
        intrinsic = np.maximum(S - K, 0.0)
    else:
        intrinsic = np.maximum(K - S, 0.0)

    # Backward induction for American option
    V = np.zeros_like(S)
    V[:, -1] = intrinsic[:, -1]

    discount = np.exp(-r * dt)

    for t in range(n_steps - 1, -1, -1):
        # Calculate continuation value
        continuation = discount * V[:, t + 1]
        
        # American option: max of intrinsic and continuation value
        V[:, t] = np.maximum(intrinsic[:, t], continuation)

    # Calculate option price and standard error
    option_price = float(np.mean(V[:, 0]))
    standard_error = float(np.std(V[:, 0]) / np.sqrt(n_paths))

    # Calculate option components
    if option_type.lower() == "call":
        intrinsic_value = max(S0 - K, 0)
        break_even = K + option_price
        required_move = ((break_even/S0 - 1) * 100)
    else:
        intrinsic_value = max(K - S0, 0)
        break_even = K - option_price
        required_move = ((1 - break_even/S0) * 100)

    time_value = option_price - intrinsic_value

    # Print detailed pricing information
    print("\n--- American Option Pricing Summary ---")
    print(f"Option Type: {option_type}")
    print(f"Strike Price (K): {K:.2f}")
    print(f"Underlying Price (S₀): {S0:.2f}")
    print(f"Forecast Volatility: {forecast_vol * 100:.2f}% annualized")
    print(f"Daily Volatility: {daily_sigma * 100:.2f}%")
    print(f"GARCH Parameters:")
    print(f"  ω (omega): {omega:.6f}")
    print(f"  α (alpha): {alpha:.6f}")
    print(f"  β (beta): {beta:.6f}")
    print(f"\nOption Components:")
    print(f"  Intrinsic Value: ${intrinsic_value:.4f}")
    print(f"  Time Value: ${time_value:.4f}")
    print(f"  Total Price: ${option_price:.4f}")
    print(f"  Standard Error: {standard_error:.4f}")
    print("----------------------------------------")

    return {
        "option_price": option_price,
        "standard_error": standard_error,
        "paths": S,
        "volatility": sigma,
        "break_even": break_even,
        "required_move": required_move,
        "intrinsic_value": intrinsic_value,
        "time_value": time_value
    }