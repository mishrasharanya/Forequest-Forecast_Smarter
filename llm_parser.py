import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv(override=True)

_client = None


def _get_api_key():
    """
    Get Groq API key from Streamlit Secrets first,
    then local .env / environment variable.
    """

    try:
        import streamlit as st

        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]

    except Exception:
        pass

    return os.getenv("GROQ_API_KEY")


def _get_client():
    """
    Create and cache Groq client.
    Works locally and on Streamlit Cloud.
    """

    global _client

    if _client is None:
        api_key = _get_api_key()

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not found. Add it in Streamlit Secrets or local .env."
            )

        if not api_key.startswith("gsk_"):
            raise RuntimeError(
                "Invalid Groq key format. Groq keys should start with 'gsk_'."
            )

        _client = Groq(api_key=api_key)

    return _client


def get_forecast_plan(user_prompt: str) -> dict:
    """
    Parses a forecasting prompt into a structured JSON plan
    using Groq-hosted Llama.
    """

    system_msg = """
    You are a financial forecasting assistant.

    Extract forecasting parameters from the user's request.

    Return ONLY valid JSON with these keys:

    {
        "model": "",
        "ticker": "",
        "start_date": "",
        "end_date": "",
        "forecast_horizon": 0
    }

    Rules:
    - model should be ARIMA, GARCH, LSTM, or another forecasting model if specified.
    - ticker should be the stock symbol.
    - dates should be yyyy-mm-dd when available.
    - forecast_horizon should be an integer number of days.
    - Return JSON only.
    """

    response = _get_client().chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": system_msg,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    output = response.choices[0].message.content.strip()

    try:
        return json.loads(output)

    except json.JSONDecodeError:
        try:
            start = output.find("{")
            end = output.rfind("}") + 1
            return json.loads(output[start:end])

        except Exception as e:
            raise ValueError(
                f"Could not parse model response as JSON:\n{output}"
            ) from e


def chat(
    prompt: str,
    model: str = "llama-3.1-8b-instant"
) -> str:
    """
    General-purpose ForeQuest chatbot.
    Used by Streamlit chatbot and option analysis.
    """

    response = _get_client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are ForeQuest AI, a financial forecasting assistant. "
                    "Help users understand volatility forecasting, GARCH models, "
                    "American option pricing, Longstaff-Schwartz Monte Carlo, "
                    "Quasi-Monte Carlo simulation, risk analysis, and stock market behavior. "
                    "Keep explanations clear, practical, and beginner-friendly."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.4,
        max_tokens=1000,
    )

    return response.choices[0].message.content.strip()