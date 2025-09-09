# Agentic AI Trading: DQN Agent for AAPL Stock

This mini-project shows how to build a simple **agent** (Agentic AI) for algorithmic trading:
1) pull historical AAPL data from Yahoo Finance →  
2) compute simple features →  
3) train a **DQN** agent in a basic trading environment →  
4) run a test pass and measure profit.

All logic lives in the notebook **`Building_an_AI_Agent_using_Agentic_AI.ipynb`**.

---

## ⚙️ What’s implemented in the notebook

**1) Data loading**
- Uses `yfinance` for **AAPL** with the range **2020-01-01…2025-02-14**:
  ```python
  symbol = "AAPL"
  start_date = "2020-01-01"
  end_date = "2025-02-14"
  data = yf.download(symbol, start=start_date, end=end_date)
**2) Feature engineering
Three simple features:
  SMA_5 — 5-day simple moving average
  SMA_20 — 20-day simple moving average
  Returns — daily return via Close.pct_change()
  ```python
  data['SMA_5'] = data['Close'].rolling(window=5).mean()
  data['SMA_20'] = data['Close'].rolling(window=20).mean()
  data['Returns'] = data['Close'].pct_change()
  data.dropna(inplace=True)
  data.reset_index(drop=True, inplace=True)
  ```
**3) Problem setup
  Actions: HOLD (0), BUY (1), SELL (2).
  State (4 numbers): [Close, SMA_5, SMA_20, Returns].


**4) Trading environment (TradingEnvironment)
  Init: starting balance $10,000, positions = 0.
  Step:
    At each bar, the agent can buy with full balance, sell all, or hold.
    Episode ends at the last bar.
    Reward: 0 during steps; at the end — balance - initial_balance (final P&L).
    Returns (next_state, reward, done, info).


**5) Model & agent
  DQN network: Linear(4→64) → ReLU → Linear(64→64) → ReLU → Linear(64→3).
  Experience replay buffer deque(maxlen=2000), gamma=0.95.
  epsilon-greedy: start 1.0, decay 0.995, min 0.01. 
  Optimizer Adam(lr=1e-3), loss MSELoss.
  Methods implemented: remember, act, replay (mini-batch training).

**6) Training
  episodes = 500, batch_size = 32.
  Each episode iterates over the whole period, stores transitions, and calls replay.

**7) Testing
  A fresh environment on the same data; the agent runs with the learned policy and prints:
  Final Balance
  Total Profit relative to $10,000

🧩 Stack
  yfinance — historical prices
  pandas, numpy — data processing
  torch — DQN model & training
  standard libs: random, collections.deque


🚀 Quick start
Option A — Google Colab / Jupyter
Install dependencies in the first cell:
```  %pip install -q yfinance pandas numpy torch```

📁 Suggested repo layout
  .
  ├── README.md
  ├── requirements.txt
  └── Building_an_AI_Agent_using_Agentic_AI.ipynb

requirements.txt:
  yfinance>=0.2.50
  pandas>=2.0.0
  numpy>=1.26.0
  torch>=2.2.0
  google colab



  🧠 Rationale & next upgrades

This notebook is a teaching demo of an Agentic loop: agent ↔ environment, replay-based learning. To make it closer to real trading and improve learning stability:
  1.Step-wise rewards
    Shape rewards with per-step equity change minus transaction costs instead of “only at the end.”
  2.Transaction costs & slippage
    Add realistic fees/spreads—they strongly affect behavior.
  3,Evaluation policy
    Before testing, fix greediness: agent.epsilon = 0.0 (no random actions).
  4.Target network / Double DQN
    Classic stability improvements for DQN.
  5.Richer features
    RSI, MACD, volatility, lagged features, calendar features.
  6.Time-based splits
    Train/val/test over non-overlapping time ranges for honest generalization.
  7.Seed control
    Set seeds for reproducibility.

    🛠 Troubleshooting

Torch install issues
  Use pip install torch for CPU; for GPU follow the official PyTorch install guide that matches your CUDA.
  yfinance returns empty data
  Try a different date range/ticker, check connection, or add retries for rate limiting.
  Varying results per run
  Normal due to randomness (epsilon-greedy, init). Fix seeds and/or increase episodes.


Nikita Marshchonok
