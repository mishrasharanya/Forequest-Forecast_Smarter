# ForeQuest — Forecast Smarter

## Overview

ForeQuest is an interactive financial analytics platform built with Streamlit that combines volatility forecasting, American option pricing, and AI-powered financial explanations into a single application.

The platform allows users to analyze publicly traded stocks, forecast future volatility using GARCH models, and price American-style options using Quasi-Monte Carlo simulation with Longstaff-Schwartz regression.

In addition, ForeQuest integrates a Groq-hosted Llama chatbot that helps users interpret forecasts, option pricing outputs, and financial concepts through natural language interactions.

---

## Features

### Volatility Forecasting

- Downloads real-time stock market data using Yahoo Finance.
- Computes daily log returns.
- Automatically selects the best GARCH(p,q) model using Akaike Information Criterion (AIC).
- Forecasts future annualized volatility.
- Visualizes historical and forecasted volatility.
- Captures volatility clustering and mean reversion dynamics.

### GARCH Diagnostics

- Q-Q Plot
- Residual Time Series Plot
- Residual Autocorrelation Function (ACF)
- Standardized Residual Analysis

### American Option Pricing

- Quasi-Monte Carlo simulation with Sobol sequences
- GARCH-driven stochastic volatility
- Longstaff-Schwartz Least Squares Monte Carlo (LSM)

### AI Financial Assistant

- Explain volatility forecasts
- Interpret option pricing outputs
- Answer financial questions
- Explain GARCH modeling
- Parse natural-language forecasting requests

---

## Installation

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

## Run

```bash
streamlit run app.py
```

## Author

Sharanya Mishra
