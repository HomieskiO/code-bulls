import backtrader as bt
from collections import defaultdict
from datetime import datetime
from stats.models import TradeLogEntry, SymbolStats, StrategyStats


class ThreeLayerAnalyzer(bt.Analyzer):
    def start(self):
        self.trades_history = []
        self.trade_id_counter = 1
        self.strategy_name = self.strategy.__class__.__name__

        # Helper dictionary to track the max size of a trade while open
        # This is needed because 'trade.size' becomes 0 when closed.
        # Key = Trade Reference, Value = Max Size seen
        self._max_sizes = {}

    def notify_trade(self, trade):
        # 1. Track trade size while it is active
        # We store the maximum absolute size seen to capture the original entry size
        if trade.ref not in self._max_sizes:
            self._max_sizes[trade.ref] = trade.size
        else:
            if abs(trade.size) > abs(self._max_sizes[trade.ref]):
                self._max_sizes[trade.ref] = trade.size

        # If trade is still open, do not log yet
        if not trade.isclosed:
            return

        # 2. Retrieve original size from memory
        original_size = self._max_sizes.pop(trade.ref, 0)

        # 3. Fix date formats (handle float vs datetime objects)
        def to_datetime(val):
            if isinstance(val, (float, int)):
                return bt.num2date(val)
            return val

        entry_dt = to_datetime(trade.open_datetime())
        exit_dt = to_datetime(trade.close_datetime())
        duration = (exit_dt - entry_dt).total_seconds() / 86400.0

        # 4. Determine side (Long/Short) based on original size
        side = 'Long' if original_size > 0 else 'Short'

        # 5. Calculations
        pnl_net = trade.pnlcomm
        entry_price = trade.price
        entry_value = abs(entry_price * original_size)
        pnl_pct = (pnl_net / entry_value) * 100 if entry_value else 0.0

        log_entry = TradeLogEntry(
            ticket_id=self.trade_id_counter,
            strategy_name=self.strategy_name,
            symbol=trade.data._name,
            side=side,
            entry_time=entry_dt.isoformat(),
            exit_time=exit_dt.isoformat(),
            entry_price=entry_price,
            exit_price=trade.data.close[0],
            size=original_size,  # Now this will use the real size, not 0
            pnl_net=pnl_net,
            pnl_pct=pnl_pct,
            duration_days=duration
        )

        self.trades_history.append(log_entry)
        self.trade_id_counter += 1

    def stop(self):
        # --- Generate Per-Symbol Report ---
        trades_by_symbol = defaultdict(list)
        for t in self.trades_history:
            trades_by_symbol[t.symbol].append(t)

        self.symbol_stats_list = []
        for symbol, trades in trades_by_symbol.items():
            total_pnl = sum(t.pnl_net for t in trades)
            wins = sum(1 for t in trades if t.pnl_net > 0)
            total_count = len(trades)

            stat = SymbolStats(
                strategy_name=self.strategy_name,
                symbol=symbol,
                total_trades=total_count,
                net_profit=total_pnl,
                win_rate=(wins / total_count * 100) if total_count else 0,
                avg_trade_pnl=total_pnl / total_count if total_count else 0,
                best_trade=max((t.pnl_net for t in trades), default=0),
                worst_trade=min((t.pnl_net for t in trades), default=0)
            )
            self.symbol_stats_list.append(stat)

        # --- Generate Strategy Summary Report ---
        # Safely retrieve data from external analyzers
        def get_an(name):
            if hasattr(self.strategy.analyzers, name):
                return getattr(self.strategy.analyzers, name).get_analysis()
            return {}

        dd_an = get_an('drawdown')
        sharpe_an = get_an('sharpe')

        all_pnl = sum(t.pnl_net for t in self.trades_history)
        all_wins = sum(1 for t in self.trades_history if t.pnl_net > 0)
        all_losses_pnl = sum(abs(t.pnl_net) for t in self.trades_history if t.pnl_net < 0)
        all_wins_pnl = sum(t.pnl_net for t in self.trades_history if t.pnl_net > 0)
        count = len(self.trades_history)

        sorted_symbols = sorted(self.symbol_stats_list, key=lambda x: x.net_profit)

        self.strategy_stats = StrategyStats(
            strategy_id=self.strategy_name,
            total_pnl=all_pnl,
            total_trades=count,
            win_rate=(all_wins / count * 100) if count else 0,
            profit_factor=(all_wins_pnl / all_losses_pnl) if all_losses_pnl > 0 else 999,
            max_drawdown_pct=dd_an.get('max', {}).get('drawdown', 0.0) if dd_an else 0,
            sharpe_ratio=sharpe_an.get('sharperatio', 0.0) if sharpe_an and sharpe_an.get('sharperatio') else 0.0,
            best_symbol=sorted_symbols[-1].symbol if sorted_symbols else "N/A",
            worst_symbol=sorted_symbols[0].symbol if sorted_symbols else "N/A"
        )

    def get_results(self):
        return {
            "log": self.trades_history,
            "by_symbol": self.symbol_stats_list,
            "global": self.strategy_stats
        }