from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# --- File 1: Trade Log (Single Operation Level) ---
@dataclass
class TradeLogEntry:
    ticket_id: int
    strategy_name: str
    symbol: str
    side: str          # 'Long' or 'Short'
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    size: float
    pnl_net: float     # Profit after commission
    pnl_pct: float     # Return percentage on this trade
    duration_days: float

# --- File 2: Symbol Performance (Asset Level) ---
@dataclass
class SymbolStats:
    strategy_name: str
    symbol: str
    total_trades: int
    net_profit: float
    win_rate: float
    avg_trade_pnl: float
    best_trade: float
    worst_trade: float

# --- File 3: Strategy/Portfolio Performance (Manager Level) ---
@dataclass
class StrategyStats:
    strategy_id: str   # Name of strategy
    total_pnl: float
    total_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    best_symbol: str   # The name of the asset that made the most money
    worst_symbol: str  # The name of the asset that lost the most money