from datetime import datetime
from typing import Any, Dict, List
import backtrader as bt


class AdvancedTradeLog(bt.Analyzer):
    """
    Analyzer that records a detailed trade log (one row per completed trade).
    Enhanced to include commissions, cumulative PnL, and bar duration.
    """

    def start(self) -> None:
        # List of completed trades (rows)
        self.trades: List[Dict[str, Any]] = []

        # Internal mapping from trade object to its "open" info
        self._open_trades: Dict[int, Dict[str, Any]] = {}

        # Incremental trade id
        self._trade_id: int = 0

        # Track cumulative PnL across all trades
        self._cumulative_pnl: float = 0.0

        # Strategy name
        self._strategy_name = self.strategy.__class__.__name__

    def notify_trade(self, trade: bt.Trade) -> None:
        """
        Called by backtrader whenever a trade is opened, updated, or closed.
        """
        data = trade.data
        if data is None:
            return

        current_dt: datetime = data.datetime.datetime(0)
        obj_key: int = id(trade)

        # --- Detect Open Trade ---
        if trade.isopen and obj_key not in self._open_trades:
            self._trade_id += 1
            entry_price: float = float(trade.price)
            size: float = float(trade.size)
            equity_at_entry: float = float(self.strategy.broker.getvalue())

            # Avoid division by zero
            if equity_at_entry != 0.0:
                position_value_at_entry = abs(entry_price * size)
                position_fraction_of_equity = position_value_at_entry / equity_at_entry
            else:
                position_value_at_entry = abs(entry_price * size)
                position_fraction_of_equity = 0.0

            side: str = "long" if size > 0 else "short"

            self._open_trades[obj_key] = {
                "trade_id": self._trade_id,
                "strategy_name": self._strategy_name,
                "data_name": getattr(data, "_name", None),
                "side": side,
                "entry_time": current_dt,
                "entry_bar_index": len(data),  # Track bar index to calculate duration later
                "entry_price": entry_price,
                "size": size,
                "equity_at_entry": equity_at_entry,
                "position_value_at_entry": position_value_at_entry,
                "position_fraction_of_equity": position_fraction_of_equity,
            }

        # --- Detect Closed Trade ---
        if trade.isclosed:
            # Retrieve opening data
            if obj_key in self._open_trades:
                open_info = self._open_trades.pop(obj_key)
            else:
                # Fallback if trade was opened before analyzer started (rare)
                return

            entry_time: datetime = open_info["entry_time"]
            exit_time: datetime = current_dt

            # PnL calculations
            # pnl = gross profit, pnlcomm = net profit (after commission)
            pnl_gross: float = float(trade.pnl)
            pnl_net: float = float(trade.pnlcomm)
            commission: float = float(trade.commission)

            # Update cumulative PnL
            self._cumulative_pnl += pnl_net

            position_value_at_entry = float(open_info["position_value_at_entry"])
            if position_value_at_entry != 0.0:
                pnl_pct: float = pnl_net / position_value_at_entry
            else:
                pnl_pct = 0.0

            # Time and Bar duration
            holding_period_days: float = 0.0
            if isinstance(entry_time, datetime) and isinstance(exit_time, datetime):
                delta = exit_time - entry_time
                holding_period_days = delta.total_seconds() / 86400.0

            bars_in_trade = len(data) - open_info["entry_bar_index"]

            # Build the row
            row: Dict[str, Any] = {
                "trade_id": open_info["trade_id"],
                "data_name": open_info["data_name"],
                "side": open_info["side"],
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "entry_price": float(open_info["entry_price"]),
                "exit_price": float(data.close[0]),
                "size": float(open_info["size"]),
                "position_value": position_value_at_entry,
                "pnl_gross": pnl_gross,
                "commission": commission,
                "pnl_net": pnl_net,
                "pnl_pct": pnl_pct,
                "cumulative_pnl": self._cumulative_pnl,
                "holding_days": holding_period_days,
                "holding_bars": bars_in_trade
            }

            self.trades.append(row)

    def get_analysis(self) -> List[Dict[str, Any]]:
        return self.trades