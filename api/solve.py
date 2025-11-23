from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import cvxpy as cp
import re, json

app = FastAPI(title="Qfolio Portfolio Solver API", version="1.0.0")

# ------------------------
#  Mocked Data (Stage 1)
# ------------------------
TICKERS = ['AAPL', 'MSFT', 'NEE', 'GOOGL', 'TSLA']
N = len(TICKERS) 
R = pd.Series({'AAPL': 0.236204, 'MSFT': 0.252517, 'NEE': 0.060919, 'GOOGL': 0.422263, 'TSLA': 0.442622})
SIGMA_NP = np.array([
    [0.069420, 0.032245, 0.011194, 0.038482, 0.072475],
    [0.032245, 0.056169, 0.004809, 0.037883, 0.054362],
    [0.011194, 0.004809, 0.074245, 0.005207, 0.010836],
    [0.038482, 0.037883, 0.005207, 0.092714, 0.071004],
    [0.072475, 0.054362, 0.010836, 0.071004, 0.379930]
])
ESG_SCORES = pd.Series({'AAPL': 16, 'MSFT': 13, 'NEE': 28, 'GOOGL': 18, 'TSLA': 28})

# ------------------------
# Request Body Definition
# ------------------------
class InputData(BaseModel):
    user_intent: str
    asset_data: str = ""

# -------------------------------------------------------
# Helper: Clean noisy JSON from LLM output (Stage 3)
# -------------------------------------------------------
def clean_llm_json(raw):
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except:
        return None

# -------------------------------------------------------
# Mock LLM ideation (Replace with Watsonx.ai later)
# -------------------------------------------------------
def mock_llm_call(intent):
    noisy = """
    Hello here is the result
    {
      "strategy_name": "Defensive Core",
      "objective": "minimize_volatility",
      "constraints": {
        "min_expected_return": 0.15,
        "max_single_asset_weight": 0.30,
        "max_esg_risk_score": 18
      },
      "reasoning_trace": "LLM reasoning summary."
    }
    EXTRA NOISE
    """
    return noisy

# -------------------------------------------------------
# Translate LLM strategy → Optimization problem (Stage 4)
# -------------------------------------------------------
def translate_strategy(strategy):
    objective_type = strategy["objective"]
    cons = strategy["constraints"]

    w = cp.Variable(N)
    variance = cp.quad_form(w, SIGMA_NP)

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= cons.get("max_single_asset_weight", 1.0)
    ]

    if "min_expected_return" in cons:
        constraints.append(R.values @ w >= cons["min_expected_return"])

    if "max_esg_risk_score" in cons:
        constraints.append(ESG_SCORES.values @ w <= cons["max_esg_risk_score"])

    if objective_type == "minimize_volatility":
        objective = cp.Minimize(variance)
    else:
        gamma = 0.5
        objective = cp.Maximize(R.values @ w - gamma * variance)

    return cp.Problem(objective, constraints), w

# -------------------------------------------------------
# Solve the optimization (Stage 5)
# -------------------------------------------------------
def solve_optimizer(problem, w):
    problem.solve()
    if w.value is None:
        return None

    weights = pd.Series(w.value, index=TICKERS).round(4)
    weights[weights < 1e-4] = 0
    if weights.sum() > 0:
        weights /= weights.sum()

    opt_return = float(weights.values @ R.values)
    opt_vol = float(np.sqrt(weights.values @ SIGMA_NP @ weights.values))
    sharpe = (opt_return - 0.02) / opt_vol

    return {
        "optimal_weights": weights.to_dict(),
        "expected_return": opt_return,
        "volatility": opt_vol,
        "sharpe_ratio": sharpe
    }

# ------------------------
# Main API Endpoint
# ------------------------
@app.post("/solve-portfolio")
def solve_portfolio(data: InputData):

    llm_output = mock_llm_call(data.user_intent)
    cleaned = clean_llm_json(llm_output)

    if not cleaned:
        return {"status": "Error", "message": "LLM produced invalid JSON"}

    problem, w = translate_strategy(cleaned)
    result = solve_optimizer(problem, w)

    return {
        "status": "Solved",
        "strategy_name": cleaned["strategy_name"],
        "llm_reasoning": cleaned["reasoning_trace"],
        **result
    }

