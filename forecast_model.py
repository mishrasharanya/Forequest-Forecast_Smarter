import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from arch import arch_model
from scipy.stats.qmc import Sobol
from scipy.stats import norm


def get_stock_data(ticker, start_date, end_date):
    """
    Fetch stock data from yfinance and calculate log returns.
    """

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data found for {ticker}")

    # yfinance sometimes returns MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        raise ValueError("Column 'Close' not found in data.")

    close_prices = df["Close"]

    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]

    close_prices = close_prices.dropna()

    if close_prices.empty:
        raise ValueError("No valid close prices found.")

    # log return = log(P_t / P_{t-1})
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()

    df = df.loc[log_returns.index].copy()
    df["LogReturns"] = log_returns.astype(float)

    return df.dropna()


def run_garch_forecast(data, forecast_horizon, ticker):
    """
    Fit a GARCH(p, q) model and forecast annualized volatility.
    """

    series = data["LogReturns"].dropna().astype(float)

    if len(series) < 30:
        raise ValueError("Not enough return data to fit GARCH model.")

    best_aic = float("inf")
    best_order = (1, 1)
    best_model = None

    for p in range(1, 4):
        for q in range(1, 4):
            try:
                model = arch_model(
                    series,
                    vol="GARCH",
                    p=p,
                    q=q,
                    dist="studentst",
                    rescale=False
                )

                fit = model.fit(
                    disp="off",
                    show_warning=False
                )

                alpha = float(fit.params.get("alpha[1]", 0.0))
                beta = float(fit.params.get("beta[1]", 0.0))

                # Stationarity condition for GARCH(1,1)-style check
                if alpha + beta < 1 and fit.aic < best_aic:
                    best_aic = fit.aic
                    best_order = (p, q)
                    best_model = fit

            except Exception:
                continue

    if best_model is None:
        raise ValueError("Could not fit a valid GARCH model.")

    hist_vol = best_model.conditional_volatility.dropna().iloc[-30:]
    hist_vol_annual = hist_vol * np.sqrt(252)

    omega = float(best_model.params.get("omega", 1e-6))
    alpha = float(best_model.params.get("alpha[1]", 0.05))
    beta = float(best_model.params.get("beta[1]", 0.90))

    if alpha + beta >= 1:
        alpha = 0.05
        beta = 0.90

    long_run_variance = omega / max(1 - alpha - beta, 1e-8)
    long_run_vol = np.sqrt(long_run_variance * 252)

    forecast_horizon = int(forecast_horizon)

    if forecast_horizon <= 0:
        raise ValueError("Forecast horizon must be greater than 0.")

    forecast_series = np.zeros(forecast_horizon)
    forecast_series[0] = float(hist_vol_annual.iloc[-1])

    for t in range(1, forecast_horizon):
        previous_daily_vol = forecast_series[t - 1] / np.sqrt(252)

        # GARCH expected forecast:
        # sigma^2_{t+1} = omega + (alpha + beta) * sigma^2_t
        variance = omega + (alpha + beta) * previous_daily_vol**2

        next_vol = np.sqrt(max(variance, 1e-10) * 252)

        # Smooth toward long-run volatility
        forecast_series[t] = 0.75 * next_vol + 0.25 * long_run_vol

    max_historical_vol = float(hist_vol_annual.max())
    min_historical_vol = float(hist_vol_annual.min())

    forecast_series = np.minimum(
        forecast_series,
        max_historical_vol * 3.5
    )

    forecast_series = np.maximum(
        forecast_series,
        min_historical_vol * 0.7
    )

    forecast_series = np.clip(
        forecast_series,
        0.05,
        1.00
    )

    plt.figure(figsize=(12, 6))

    hist_x = np.arange(len(hist_vol_annual))
    forecast_x = np.arange(
        len(hist_vol_annual) - 1,
        len(hist_vol_annual) + forecast_horizon
    )

    plt.plot(
        hist_x,
        hist_vol_annual.values,
        label="Historical Volatility",
        linewidth=2
    )

    plt.plot(
        forecast_x,
        np.concatenate([[hist_vol_annual.iloc[-1]], forecast_series]),
        label=f"GARCH{best_order} Forecast",
        linewidth=2,
        marker="o"
    )

    plt.axvline(
        x=len(hist_vol_annual) - 1,
        linestyle="--",
        linewidth=1
    )

    plt.title(f"{ticker} GARCH{best_order} Volatility Forecast")
    plt.xlabel("Days")
    plt.ylabel("Annualized Volatility")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("garch_forecast_plot.png", dpi=300)
    plt.close()

    plot_garch_diagnostics(best_model)

    return forecast_series, best_order, best_model.params


def plot_garch_diagnostics(model_fit):
    """
    Save residual diagnostic plots for the fitted GARCH model.
    """

    residuals = (
        model_fit.resid / model_fit.conditional_volatility
    ).dropna()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sm.qqplot(residuals, line="s", ax=axes[0])
    axes[0].set_title("GARCH Q-Q Plot")

    plot_acf(residuals, lags=40, ax=axes[1])
    axes[1].set_title("GARCH Residual ACF")

    axes[2].plot(range(len(residuals)), residuals)
    axes[2].axhline(0, linestyle="--")
    axes[2].set_title("GARCH Residual Time Plot")

    plt.tight_layout()
    plt.savefig("garch_diagnostics.png", dpi=300)
    plt.close()


def price_american_option_qmc(
    S0,
    K,
    r,
    T,
    n_paths,
    option_type,
    garch_params,
    forecast_vol
):
    """
    Price an American option using:
    1. Quasi-Monte Carlo simulation with Sobol sequences
    2. GARCH-driven time-varying volatility
    3. Longstaff-Schwartz regression for early exercise

    American option formula:
    V_t = max(exercise value, expected discounted continuation value)
    """

    S0 = float(S0)
    K = float(K)
    r = float(r)
    T = float(T)
    n_paths = int(n_paths)
    forecast_vol = float(forecast_vol)

    if S0 <= 0:
        raise ValueError("Current stock price must be greater than 0.")

    if K <= 0:
        raise ValueError("Strike price must be greater than 0.")

    if T <= 0:
        raise ValueError("Time to expiry must be greater than 0.")

    if n_paths <= 0:
        raise ValueError("Number of paths must be greater than 0.")

    if forecast_vol <= 0:
        raise ValueError("Forecast volatility must be greater than 0.")

    option_type = option_type.lower()

    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be either 'call' or 'put'.")

    n_steps = max(int(T * 252), 1)
    dt = T / n_steps

    daily_sigma = forecast_vol / np.sqrt(252)

    omega = float(garch_params.get("omega", 1e-6))
    alpha = float(garch_params.get("alpha[1]", 0.05))
    beta = float(garch_params.get("beta[1]", 0.90))

    sobol = Sobol(
        d=n_steps,
        scramble=True,
        seed=42
    )

    u = sobol.random(n_paths)
    u = np.clip(u, 1e-10, 1 - 1e-10)
    z = norm.ppf(u)

    S = np.zeros((n_paths, n_steps + 1))
    sigma = np.zeros_like(S)
    log_returns = np.zeros_like(S)

    S[:, 0] = S0
    sigma[:, 0] = daily_sigma

    for t in range(1, n_steps + 1):
        z_t = z[:, t - 1]

        drift = (r - 0.5 * sigma[:, t - 1] ** 2) * dt
        diffusion = sigma[:, t - 1] * np.sqrt(dt) * z_t

        log_returns[:, t] = drift + diffusion
        S[:, t] = S[:, t - 1] * np.exp(log_returns[:, t])

        shock_sq = log_returns[:, t] ** 2

        variance = (
            omega
            + alpha * shock_sq
            + beta * sigma[:, t - 1] ** 2
        )

        sigma[:, t] = np.sqrt(
            np.maximum(variance, 1e-10)
        )

        sigma[:, t] = np.clip(
            sigma[:, t],
            daily_sigma * 0.5,
            daily_sigma * 2.0
        )

        S[:, t] = np.clip(
            S[:, t],
            S0 * 0.1,
            S0 * 10.0
        )

    if option_type == "call":
        intrinsic = np.maximum(S - K, 0.0)
    else:
        intrinsic = np.maximum(K - S, 0.0)

    # Longstaff-Schwartz Monte Carlo
    cashflows = intrinsic[:, -1].copy()
    exercise_time = np.full(n_paths, n_steps)

    for t in range(n_steps - 1, 0, -1):
        in_the_money = intrinsic[:, t] > 0

        if np.sum(in_the_money) < 5:
            continue

        X = S[in_the_money, t]

        Y = cashflows[in_the_money] * np.exp(
            -r * dt * (exercise_time[in_the_money] - t)
        )

        regression_matrix = np.column_stack([
            np.ones_like(X),
            X,
            X ** 2
        ])

        coeffs = np.linalg.lstsq(
            regression_matrix,
            Y,
            rcond=None
        )[0]

        continuation_value = regression_matrix @ coeffs
        exercise_value = intrinsic[in_the_money, t]

        should_exercise = exercise_value > continuation_value

        itm_indices = np.where(in_the_money)[0]
        exercise_indices = itm_indices[should_exercise]

        cashflows[exercise_indices] = intrinsic[exercise_indices, t]
        exercise_time[exercise_indices] = t

    discounted_cashflows = cashflows * np.exp(
        -r * dt * exercise_time
    )

    option_price = float(np.mean(discounted_cashflows))

    standard_error = float(
        np.std(discounted_cashflows, ddof=1) / np.sqrt(n_paths)
    )

    if option_type == "call":
        intrinsic_value = max(S0 - K, 0)
        break_even = K + option_price
        required_move = ((break_even / S0 - 1) * 100)
    else:
        intrinsic_value = max(K - S0, 0)
        break_even = K - option_price
        required_move = ((1 - break_even / S0) * 100)

    time_value = option_price - intrinsic_value

    return {
        "option_price": float(option_price),
        "standard_error": float(standard_error),
        "paths": S,
        "volatility": sigma,
        "break_even": float(break_even),
        "required_move": float(required_move),
        "intrinsic_value": float(intrinsic_value),
        "time_value": float(time_value),
        "exercise_time": exercise_time,
    }