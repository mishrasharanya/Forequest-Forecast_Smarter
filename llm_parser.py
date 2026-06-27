import os
import json
from dotenv import load_dotenv
from groq import Groq

# Force .env values to override existing environment variables
load_dotenv(override=True)

_client = None


def _get_client():
    """
    Create and cache a Groq client.
    """
    global _client

    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not found. Check your .env file."
            )

        print(f"Using API key prefix: {api_key[:8]}")

        if not api_key.startswith("gsk_"):
            raise RuntimeError(
                f"Invalid Groq key format. Expected key starting with 'gsk_' but got '{api_key[:8]}...'"
            )

        _client = Groq(api_key=api_key)

    return _client


def get_forecast_plan(user_prompt: str) -> dict:
    """
    Convert a natural-language forecasting request
    into a structured JSON plan.
    """

    system_msg = """
    You are a financial forecasting assistant.

    Extract forecasting parameters from the user's request.

    Return ONLY valid JSON:

    {
        "model": "",
        "ticker": "",
        "start_date": "",
        "end_date": "",
        "forecast_horizon": 0
    }

    Rules:
    - model should be ARIMA, GARCH, LSTM, etc.
    - ticker should be the stock ticker.
    - dates must be yyyy-mm-dd when available.
    - forecast_horizon must be an integer.
    - Return JSON only.
    """

    response = _get_client().chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
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
                f"Could not parse JSON response:\n{output}"
            ) from e


def chat(
    prompt: str,
    model: str = "llama-3.1-8b-instant"
) -> str:
    """
    General-purpose ForeQuest chatbot.
    """

    response = _get_client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are ForeQuest AI, a financial forecasting assistant. "
                    "Help users understand volatility forecasting, stock prices, "
                    "risk analysis, option pricing, GARCH models, ARIMA models, "
                    "and financial markets."
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