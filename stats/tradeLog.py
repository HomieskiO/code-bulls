
from datetime import datetime
from typing import Any, Dict, List

import backtrader as bt


class TradeLog(bt.Analyzer):
    """
    Analyzer that records a detailed trade log (one row per completed trade).

    Each trade row includes:
      - trade_id: incremental internal id
      - data_name: name of the data feed (symbol)
      - side: 'long' or 'short'
      - entry_time, exit_time: ISO-formatted datetimes
      - entry_price, exit_price
      - size: number of units (positive = long, negative = short)
      - position_value_at_entry: abs(entry_price * size)
      - position_fraction_of_equity: position_value_at_entry / equity_at_entry
      - pnl_abs: profit/loss in currency (after commission if available)
      - pnl_pct: profit/loss as a fraction of position_value_at_entry
      - holding_period_days: (exit_time - entry_time) in days (float)
    """

    def start(self) -> None:
        # List of completed trades (rows)
        self.trades: List[Dict[str, Any]] = []

        # Internal mapping from trade object to its "open" info
        self._open_trades: Dict[int, Dict[str, Any]] = {}

        # Incremental trade id
        self._trade_id: int = 0

        #strategy name
        self._strategy_name = self.strategy.__class__.__name__


    def notify_trade(self, trade: bt.Trade) -> None:
        """
        Called by backtrader whenever a trade is opened, updated, or closed.
        We use this to detect open/close events and build a per-trade record.
        """
        # We only care about trades that have a data feed
        data = trade.data
        if data is None:
            return

        # Current datetime of the bar
        current_dt: datetime = data.datetime.datetime(0)
        obj_key: int = id(trade)

        # Detect a new/open trade (not seen before and currently open)
        if trade.isopen and obj_key not in self._open_trades:
            self._trade_id += 1

            entry_price: float = float(trade.price)
            size: float = float(trade.size)
            equity_at_entry: float = float(self.strategy.broker.getvalue())

            if equity_at_entry != 0.0:
                position_value_at_entry: float = abs(entry_price * size)
                position_fraction_of_equity: float = position_value_at_entry / equity_at_entry
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
                "entry_price": entry_price,
                "size": size,
                "equity_at_entry": equity_at_entry,
                "position_value_at_entry": position_value_at_entry,
                "position_fraction_of_equity": position_fraction_of_equity,
            }

        # Detect a closed trade
        if trade.isclosed:
            # If we don't have an open record for some reason, create a minimal one
            if obj_key not in self._open_trades:
                self._trade_id += 1
                size = float(trade.size)
                entry_price = float(trade.price)
                equity_at_entry = float(self.strategy.broker.getvalue())
                if equity_at_entry != 0.0:
                    position_value_at_entry = abs(entry_price * size)
                    position_fraction_of_equity = position_value_at_entry / equity_at_entry
                else:
                    position_value_at_entry = abs(entry_price * size)
                    position_fraction_of_equity = 0.0

                side = "long" if size > 0 else "short"

                self._open_trades[obj_key] = {
                    "trade_id": self._trade_id,
                    "data_name": getattr(data, "_name", None),
                    "side": side,
                    "entry_time": current_dt,
                    "entry_price": entry_price,
                    "size": size,
                    "equity_at_entry": equity_at_entry,
                    "position_value_at_entry": position_value_at_entry,
                    "position_fraction_of_equity": position_fraction_of_equity,
                }

            open_info: Dict[str, Any] = self._open_trades.pop(obj_key)

            entry_time: datetime = open_info["entry_time"]
            exit_time: datetime = current_dt

            # We use current market price as "exit_price"
            exit_price: float = float(data.close[0])

            # PnL: prefer pnlcomm if available, fallback to pnl
            pnl_abs: float = float(getattr(trade, "pnlcomm", trade.pnl))
            position_value_at_entry = float(open_info["position_value_at_entry"])

            if position_value_at_entry != 0.0:
                pnl_pct: float = pnl_abs / position_value_at_entry
            else:
                pnl_pct = 0.0

            # Holding period
            holding_period_days: float = 0.0
            if isinstance(entry_time, datetime) and isinstance(exit_time, datetime):
                delta = exit_time - entry_time
                holding_period_days = delta.total_seconds() / 86400.0

            row: Dict[str, Any] = {
                "trade_id": open_info["trade_id"],
                "data_name": open_info["data_name"],
                "side": open_info["side"],
                "entry_time": entry_time.isoformat() if isinstance(entry_time, datetime) else None,
                "exit_time": exit_time.isoformat() if isinstance(exit_time, datetime) else None,
                "entry_price": float(open_info["entry_price"]),
                "exit_price": exit_price,
                "size": float(open_info["size"]),
                "position_value_at_entry": position_value_at_entry,
                "position_fraction_of_equity": float(open_info["position_fraction_of_equity"]),
                "pnl_abs": pnl_abs,
                "pnl_pct": pnl_pct,
                "holding_period_days": holding_period_days,
            }

            self.trades.append(row)

    def get_analysis(self) -> List[Dict[str, Any]]:
        """
        Backtrader will call this at the end of the run.
        We return the list of trade rows collected.
        """
        return self.trades
