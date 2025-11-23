# Qfolio Portfolio Optimization API

Vercel Serverless FastAPI for portfolio optimization with IBM Watsonx.ai integration.

## 🚀 Quick Deploy to Vercel

### Step 1: Deploy via Vercel Dashboard
1. Go to [https://vercel.com/new](https://vercel.com/new)
2. Import your GitHub repository: `ashihams/Portfolio-Optimization`
3. Vercel will auto-detect the Python configuration
4. Click **Deploy**

### Step 2: Your API is Live!
Your API will be available at:
```
https://qfolio-api.vercel.app/solve-portfolio
```

## 📡 API Endpoint

**POST** `/solve-portfolio`

**Request Body:**
```json
{
  "user_intent": "Generate a defensive portfolio with ESG < 18",
  "asset_data": ""
}
```

**Response:**
```json
{
  "status": "Solved",
  "strategy_name": "Defensive Core",
  "llm_reasoning": "LLM reasoning summary.",
  "optimal_weights": {
    "AAPL": 0.25,
    "MSFT": 0.30,
    "NEE": 0.20,
    "GOOGL": 0.15,
    "TSLA": 0.10
  },
  "expected_return": 0.18,
  "volatility": 0.12,
  "sharpe_ratio": 1.33
}
```

## 🔗 Connect to IBM Orchestrate

### Option 1: Auto-generated OpenAPI
1. Go to IBM Orchestrate → Agents → Your Portfolio-Orchestrator-Agent
2. Toolset → Add tool → OpenAPI
3. Enter: `https://qfolio-api.vercel.app/openapi.json`

### Option 2: Static OpenAPI File
1. Use the `openapi.yaml` file in this repo
2. Upload it when adding the OpenAPI tool in IBM Orchestrate

## 📁 Project Structure

```
qfolio-api/
├── api/
│   └── solve.py          # FastAPI serverless endpoint
├── requirements.txt      # Python dependencies
├── vercel.json          # Vercel configuration
├── openapi.yaml         # OpenAPI spec for IBM Orchestrate
└── README.md            # This file
```

## 🛠️ Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn api.solve:app --reload
```

## 📝 Features

- ✅ Zero-config Vercel deployment
- ✅ Serverless scaling
- ✅ Free hosting tier
- ✅ Automatic OpenAPI spec generation
- ✅ IBM Orchestrate compatible
- ✅ Portfolio optimization with CVXPY
- ✅ Mock LLM integration (ready for Watsonx.ai)

## 🔄 Next Steps

1. Replace `mock_llm_call()` with actual Watsonx.ai integration
2. Add real-time asset data fetching
3. Implement additional optimization strategies

