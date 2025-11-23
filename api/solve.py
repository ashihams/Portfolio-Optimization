from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from scipy.optimize import minimize
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
    """Convert strategy to scipy.optimize constraints and objective"""
    objective_type = strategy["objective"]
    cons = strategy["constraints"]
    
    # Build constraints list for scipy.optimize
    constraints_list = []
    
    # Constraint: sum of weights = 1
    constraints_list.append({
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1.0
    })
    
    # Constraint: min expected return (if specified)
    if "min_expected_return" in cons:
        min_return = cons["min_expected_return"]
        constraints_list.append({
            'type': 'ineq',
            'fun': lambda w: np.dot(R.values, w) - min_return
        })
    
    # Constraint: max ESG risk score (if specified)
    if "max_esg_risk_score" in cons:
        max_esg = cons["max_esg_risk_score"]
        constraints_list.append({
            'type': 'ineq',
            'fun': lambda w: max_esg - np.dot(ESG_SCORES.values, w)
        })
    
    # Bounds: 0 <= w <= max_single_asset_weight
    max_weight = cons.get("max_single_asset_weight", 1.0)
    bounds = [(0, max_weight) for _ in range(N)]
    
    # Objective function
    if objective_type == "minimize_volatility":
        def objective(w):
            return np.dot(w, np.dot(SIGMA_NP, w))
    else:
        # Maximize Sharpe-like: return - gamma * variance
        gamma = 0.5
        def objective(w):
            portfolio_return = np.dot(R.values, w)
            portfolio_variance = np.dot(w, np.dot(SIGMA_NP, w))
            return -(portfolio_return - gamma * portfolio_variance)
    
    return objective, constraints_list, bounds

# -------------------------------------------------------
# Solve the optimization (Stage 5)
# -------------------------------------------------------
def solve_optimizer(objective, constraints_list, bounds):
    """Solve using scipy.optimize.minimize"""
    # Initial guess: equal weights
    x0 = np.ones(N) / N
    
    # Solve the optimization problem
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints_list,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    if not result.success:
        return None
    
    weights = pd.Series(result.x, index=TICKERS).round(4)
    weights[weights < 1e-4] = 0
    if weights.sum() > 0:
        weights /= weights.sum()

    opt_return = float(weights.values @ R.values)
    opt_vol = float(np.sqrt(weights.values @ SIGMA_NP @ weights.values))
    sharpe = (opt_return - 0.02) / opt_vol if opt_vol > 0 else 0.0

    return {
        "optimal_weights": weights.to_dict(),
        "expected_return": opt_return,
        "volatility": opt_vol,
        "sharpe_ratio": sharpe
    }

# ------------------------
# Root Endpoint
# ------------------------
@app.get("/")
def root():
    return {
        "message": "Qfolio Portfolio Solver API",
        "version": "1.0.0",
        "endpoints": {
            "solve_portfolio": "/solve-portfolio (POST)",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
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

    objective, constraints_list, bounds = translate_strategy(cleaned)
    result = solve_optimizer(objective, constraints_list, bounds)
    
    if result is None:
        return {"status": "Error", "message": "Optimization failed - constraints may be infeasible"}

    return {
        "status": "Solved",
        "strategy_name": cleaned["strategy_name"],
        "llm_reasoning": cleaned["reasoning_trace"],
        **result
    }

