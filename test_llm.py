# test_llm.py
from llm_parser import get_forecast_plan

user_input = "Forecast TSLA stock using GARCH from 2021 to 2023 for 1 month"
plan = get_forecast_plan(user_input)

print("Structured Forecast Plan:")
print(plan)