import subprocess
import json

def get_forecast_plan(user_prompt):
    """
    Calls the local Mistral model using Ollama to parse a forecasting prompt
    into a structured JSON plan.
    """
    system_instruction = f"""
You are a helpful assistant that extracts forecasting parameters from user requests.

Given a prompt like: 
    "Forecast AAPL using ARIMA from 2020 to 2023, next 5 days"

Return a valid JSON with the following keys:
- model (e.g. ARIMA, GARCH)
- ticker (e.g. AAPL)
- start_date (yyyy-mm-dd)
- end_date (yyyy-mm-dd)
- forecast_horizon (integer)

Only return valid JSON. Do not explain.
Prompt: {user_prompt}
"""

    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=system_instruction.encode(),
        stdout=subprocess.PIPE
    )

    output = result.stdout.decode().strip()

    try:
        # Try parsing directly if it’s already a JSON string
        parsed = json.loads(output)
    except:
        # Try to extract a JSON block from extra text
        start = output.find("{")
        end = output.rfind("}") + 1
        json_str = output[start:end]
        parsed = json.loads(json_str)

    return parsed
