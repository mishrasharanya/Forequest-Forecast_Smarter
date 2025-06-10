import streamlit as st
import datetime
from dotenv import load_dotenv
import os
import yfinance as yf
import numpy as np
import pandas as pd

# === Forecast Imports ===
from forecast_model import get_stock_data, run_garch_forecast, price_american_option_qmc

# === LangChain for Chatbot ===
from langchain_community.llms import Ollama

# === Load environment variables ===
load_dotenv()
model_name = os.getenv("LLAMA_MODEL", "llama3")
llm = Ollama(model=model_name)

# === Helper Functions ===
def name_to_ticker(name):
    search = yf.Ticker(name)
    try:
        info = search.info
        if 'symbol' in info:
            return info['symbol']
    except:
        return name.upper()
    return None

def get_date_range(option):
    today = datetime.date.today()
    if option == "Last 1 Year":
        return today - datetime.timedelta(days=365), today
    elif option == "Last 6 Months":
        return today - datetime.timedelta(days=182), today
    elif option == "Last 3 Months":
        return today - datetime.timedelta(days=91), today
    else:
        return today - datetime.timedelta(days=365), today

# === Streamlit Layout ===
st.set_page_config(page_title="ForeQuest", layout="wide")
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose App Section", ["🧠 Wanna Chatbot?", "📈 ForeQuest"])

# === Mode 1: Chatbot ===
if app_mode == "🧠 Wanna Chatbot?":
    st.title("🤖 Chatbot")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    prompt = st.chat_input("Type your question...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get LLM response
        response = llm.invoke(prompt)
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# === Mode 2: Forecast and Option Pricing ===
else:
    st.title("📊 Volatility Forecast & American Option Pricing")

    # Initialize session state for forecast results
    if 'forecast_results' not in st.session_state:
        st.session_state.forecast_results = None
        st.session_state.current_price = None
        st.session_state.forecast_vol = None
        st.session_state.garch_params = None

    # Forecast Section
    st.subheader("Step 1: Run Volatility Forecast")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        company_name = st.text_input("Enter Company Name (e.g., AAPL, MSFT)", value="AAPL")
        date_range_option = st.selectbox("Select Date Range", ["Last 1 Year", "Last 6 Months", "Last 3 Months"])
    
    with col2:
        forecast_horizon = st.slider("Forecast Horizon (days)", 0, 30, 7)
        if st.button("Run Forecast"):
            with st.spinner("Getting forecast model and plotting using garch model..."):
                try:
                    ticker = name_to_ticker(company_name)
                    start_date, end_date = get_date_range(date_range_option)
                    data = get_stock_data(ticker, str(start_date), str(end_date))
                    forecast, best_order, garch_params = run_garch_forecast(data, forecast_horizon, ticker)
                    
                    # Store results in session state
                    st.session_state.forecast_results = {
                        'ticker': ticker,
                        'best_order': best_order,
                        'garch_params': garch_params
                    }
                    st.session_state.current_price = float(data['Close'].iloc[-1])
                    st.session_state.forecast_vol = np.mean(forecast) / 100
                    st.session_state.garch_params = garch_params
                    
                    st.success("✅ Forecast Complete!")
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")

    # Display forecast results if available
    if st.session_state.forecast_results:
        st.subheader("📊 GARCH Model:")
        st.markdown(f"""
        - **Model Type**: GARCH({st.session_state.forecast_results['best_order'][0]},{st.session_state.forecast_results['best_order'][1]})
        - **Parameters**:
            - ω (omega): {st.session_state.forecast_results['garch_params']['omega']:.6f}
            - α (alpha): {st.session_state.forecast_results['garch_params']['alpha[1]']:.6f}
            - β (beta): {st.session_state.forecast_results['garch_params']['beta[1]']:.6f}
        """)

        st.subheader(f"📈 Volatility Forecast for {st.session_state.forecast_results['ticker']}")
        st.image("garch_forecast_plot.png", caption="GARCH Forecast")

        # Option Pricing Section
        st.subheader("Step 2: Price American Option")
        st.markdown(f"**Current Stock Price**: ${st.session_state.current_price:.2f}")
        st.markdown(f"**Forecast Volatility**: {st.session_state.forecast_vol:.2%}")

        # Option parameters
        col1, col2 = st.columns(2)
        with col1:
            K = st.number_input("Strike Price", value=round(st.session_state.current_price, 2))
            r = st.number_input("Risk-Free Rate", value=0.05)
            T = st.slider("Time to Maturity (in years)", 0.1, 2.0, 0.5)
        
        with col2:
            option_type = st.selectbox("Option Type", ["call", "put"])
            user_option_price = st.number_input("Enter Option Price", value=0.0, step=0.01, format="%.2f")

        if st.button("Price Option"):
            with st.spinner("Calculating option price..."):
                result = price_american_option_qmc(
                    S0=st.session_state.current_price,
                    K=K,
                    r=r,
                    T=T,
                    n_paths=512,  # Fixed number of paths
                    option_type=option_type,
                    garch_params=st.session_state.garch_params,
                    forecast_vol=st.session_state.forecast_vol
                )
                
                st.success("✅ Option Pricing Complete")
                
                # Calculate profit/loss scenarios
                current_price = st.session_state.current_price
                price_changes = [-0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1]  # -10% to +10%
                scenarios = []
                
                for change in price_changes:
                    new_price = current_price * (1 + change)
                    if option_type.lower() == "call":
                        intrinsic = max(new_price - K, 0)
                    else:
                        intrinsic = max(K - new_price, 0)
                    profit_loss = intrinsic - user_option_price
                    scenarios.append({
                        "price_change": change * 100,
                        "new_price": new_price,
                        "intrinsic_value": intrinsic,
                        "profit_loss": profit_loss
                    })

                st.markdown(f"""
                ###Option Analysis
                - **Option Type**: `{option_type}`
                - **Strike Price (K)**: `{K}`
                - **Underlying Price (S₀)**: `{current_price:.2f}`
                - **Your Option Price**: **${user_option_price:.2f}**
                - **Theoretical Price**: `${result['option_price']:.4f}`
                - **Forecast Volatility**: `{st.session_state.forecast_vol:.2%}`
                """)

                # Get moneyness based on option type
                if option_type.lower() == "call":
                    moneyness = "In-the-Money" if current_price > K else "Out-of-the-Money"
                else:
                    moneyness = "In-the-Money" if current_price < K else "Out-of-the-Money"

                st.markdown(f"""
                ### 📊 Profit/Loss Scenarios
                """)
                
                # Create a DataFrame for better table formatting
                scenarios_df = pd.DataFrame(scenarios)
                scenarios_df.columns = ['Price Change (%)', 'New Stock Price ($)', 'Intrinsic Value ($)', 'Profit/Loss ($)']
                scenarios_df['Price Change (%)'] = scenarios_df['Price Change (%)'].map('{:+.1f}%'.format)
                scenarios_df['New Stock Price ($)'] = scenarios_df['New Stock Price ($)'].map('${:.2f}'.format)
                scenarios_df['Intrinsic Value ($)'] = scenarios_df['Intrinsic Value ($)'].map('${:.2f}'.format)
                scenarios_df['Profit/Loss ($)'] = scenarios_df['Profit/Loss ($)'].map('${:+.2f}'.format)
                
                # Style the DataFrame
                def highlight_profit_loss(val):
                    try:
                        value = float(val.replace('$', '').replace(',', ''))
                        color = 'green' if value > 0 else 'red' if value < 0 else 'white'
                        return f'background-color: {color}; color: white; font-weight: bold'
                    except:
                        return ''

                styled_df = scenarios_df.style.applymap(highlight_profit_loss, subset=['Profit/Loss ($)'])
                st.dataframe(styled_df, use_container_width=True)

                st.markdown(f"""
                ### 📈 Break-even Analysis
                - **Moneyness**: `{moneyness}`
                - **Break-even Price**: `${result['break_even']:.2f}`
                - **Required Stock Move**: `{result['required_move']:.2f}%`
                - **Time to Expiration**: `{T:.2f} years`
                - **Intrinsic Value**: `${result['intrinsic_value']:.4f}`
                - **Time Value**: `${result['time_value']:.4f}`
                """)

                # Option Analysis using Llama
                st.subheader("Option Analysis")
                analysis_prompt = f"""
                Analyze this American {option_type} option:
                - Current Stock Price: ${current_price:.2f}
                - Strike Price: ${K:.2f}
                - Your Option Price: ${user_option_price:.2f}
                - Theoretical Price: ${result['option_price']:.4f}
                - Time to Maturity: {T:.2f} years
                - Forecast Volatility: {st.session_state.forecast_vol:.2%}
                - Risk-Free Rate: {r:.2%}
                - Break-even Price: ${result['break_even']:.2f}
                - Required Stock Move: {result['required_move']:.2f}%
                - Moneyness: {moneyness}
                - Intrinsic Value: ${result['intrinsic_value']:.4f}
                - Time Value: ${result['time_value']:.4f}

                Provide insights on:
                1. Comparison of your option price vs theoretical price
                2. Moneyness of the option and its implications
                3. Impact of volatility on the price
                4. Time value component and time decay
                5. Risk factors to consider
                6. Break-even analysis and required stock movement
                7. Probability of profit based on the break-even point

                Keep the analysis correct and in-depth but concise. Also add a summary paragraph at the end stating whether the option
                can be profitable or not, considering your option price and the required stock movement.
                """
                
                with st.spinner("Generating analysis..."):
                    analysis = llm.invoke(analysis_prompt)
                    st.markdown(analysis)
