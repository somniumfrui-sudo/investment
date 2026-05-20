"""RSIルールのシンプルバックテスト（backtrader）。"""

from __future__ import annotations

import backtrader as bt
import pandas as pd
import yfinance as yf


class RSIStrategy(bt.Strategy):
    params = (
        ("rsi_period", 14),
        ("oversold", 35),
        ("overbought", 65),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)

    def next(self):
        if self.rsi[0] is None or pd.isna(self.rsi[0]):
            return
        if not self.position:
            if float(self.rsi[0]) < self.p.oversold:
                self.buy()
        else:
            if float(self.rsi[0]) > self.p.overbought:
                self.sell()


def run_backtest(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    df = t.history(period="1y", auto_adjust=True)
    if df is None or df.empty or len(df) < 60:
        return {
            "signal": "neutral",
            "win_rate_pct": None,
            "total_trades": 0,
            "max_drawdown_pct": None,
            "error": "1年分のデータが不足しています。",
        }

    data_df = df.copy()
    data_df.columns = [c.lower() for c in data_df.columns]
    if "open" not in data_df.columns:
        return {"signal": "neutral", "error": "OHLCV列が不正です。", "total_trades": 0}

    data_df = data_df[["open", "high", "low", "close", "volume"]]
    data_df = data_df.dropna()

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100_000.0)
    cerebro.broker.setcommission(commission=0.0)

    data0 = bt.feeds.PandasData(dataname=data_df)
    cerebro.adddata(data0)
    cerebro.addstrategy(RSIStrategy)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

    results = cerebro.run()
    strat = results[0]

    ta = strat.analyzers.trades.get_analysis()
    dd = strat.analyzers.drawdown.get_analysis()

    closed = int(ta.get("total", {}).get("closed", 0) or 0)
    won = int(ta.get("won", {}).get("total", 0) or 0)

    win_rate_pct = (won / closed * 100.0) if closed > 0 else None

    max_dd = dd.get("max", {})
    max_drawdown_pct = float(max_dd.get("drawdown", 0.0) or 0.0)

    if closed == 0 or win_rate_pct is None:
        signal = "neutral"
    elif win_rate_pct >= 55.0:
        signal = "buy"
    elif win_rate_pct <= 45.0:
        signal = "sell"
    else:
        signal = "neutral"

    return {
        "signal": signal,
        "win_rate_pct": win_rate_pct,
        "total_trades": closed,
        "wins": won,
        "max_drawdown_pct": max_drawdown_pct,
        "final_value": float(cerebro.broker.getvalue()),
        "reason": "RSI<35で買い / RSI>65で売り（1年・日足）",
    }
