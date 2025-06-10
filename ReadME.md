# ForeQuest- Forecast Smarter

##  Description

ForeQuest is a Streamlit-based application that empowers users to forecast financial market volatility and perform American option pricing with real data. By selecting a company and a timeframe (last 3, 6 months, or 1 year), users can visualize historical and forecasted annualized volatility, gaining insights into market patterns such as volatility clustering and mean reversion.

Project Highlights
- Volatility Forecasting: Uses the GARCH model to forecast volatility, visualizing both historical and predicted annualized volatility.

- Model Diagnostics: Incorporates best practices from time series analysis, including Q-Q plots, residual plots, ACF plots, and the Ljung-Box test to ensure model validity.

- Option Pricing: Implements American option pricing using Quasi Monte Carlo simulations, providing detailed summaries (intrinsic value, time value, break-even, and required price movement).

- Interpretability: Designed for clarity and educational value, making complex financial modeling accessible and actionable.

## Project Motivation
ForeQuest was built to deepen understanding of financial modeling, volatility dynamics, and real-world risk quantification using Python. The tool is designed to be both interpretable and practical, providing valuable insights for anyone interested in market risk and derivatives.

## Methodology
Model Validation:
Before finalizing forecasts, the model was validated using Q-Q plots (normality of residuals), residual plots (checking for heteroscedasticity), ACF plots (autocorrelation), and the Ljung-Box test (independence of residuals). All diagnostics indicated a well-specified model.

Option Pricing:
American options are priced using Quasi Monte Carlo simulations, with comprehensive outputs for decision support.

##  Requirements

To run the Python scripts in this repository, you'll need the following:

- Python 3.8+
- Packages:
  - `numpy`
  - `matplotlib`
  - `pandas`
  - `scipy`
  - `ezc3d` (for loading `.c3d` motion capture files) 

Install them using:

```bash
pip install numpy matplotlib pandas scipy ezc3d
```

## File Description:

#### 1. app.py
Main Streamlit app. Integrates the forecasting component and provides the user interface.

#### 2.forecast_model.py
Contains core logic for volatility forecasting (GARCH) and American option pricing.

#### 3. llm_parser.py
Parses text using large language models (LLMs) for enhanced interpretability or user queries.

#### 4. test_forecast.py
Unit tests for forecast_model.py, including validation of plots and forecast accuracy.

#### 5. test_llm.py
Unit tests for llm_parser.py, ensuring the LLM integration functions as expected.


#### Reproducing Results:
1. install dependencies:
```bash
pip install streamlit numpy pandas matplotlib arch
```

2. Run to start app:
```bash
streamlit run app.py
```

3. Unit Testing:
```bash
python unittest test_forecast.py
python unittest test_llm.py
```

