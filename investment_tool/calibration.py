"""銘柄別 BUY 閾値のキャリブレーションと SELL 抑制ルール。"""

from __future__ import annotations

import pandas as pd


def ma20_is_uptrend(close: pd.Series, *, pos: int = -1, lookback: int = 5) -> bool:
    """終値系列上で、指定位置の MA20 が lookback 営業日前より上なら右肩上がりとみなす。"""
    c = close.astype(float)
    if len(c) < 20 + lookback + 1:
        return False
    ma = c.rolling(20).mean()
    i = pos if pos >= 0 else len(c) + pos
    j = i - lookback
    if j < 0:
        return False
    try:
        return float(ma.iloc[i]) > float(ma.iloc[j])
    except Exception:
        return False


def calibrate_buy_threshold(
    rows: list[dict],
    p_min: int = 50,
    p_max: int = 75,
    min_buys: int = 5,
) -> tuple[float, float | None, int]:
    """
    ウォークフォワード等の行（up_probability, actual_up）から BUY 閾値をグリッドサーチ。
    BUY 正解率が最大になる閾値を採用。同率なら閾値が高い方（保守的）を採用。
    戻り値: (閾値 0〜1, その閾値での BUY 正解率, その閾値での BUY 件数)
    """
    if not rows:
        return 0.60, None, 0

    best_t = 0.60
    best_acc = -1.0
    best_n = 0

    for pct in range(p_min, p_max + 1):
        t = pct / 100.0
        buys = [r for r in rows if float(r["up_probability"]) >= t]
        n = len(buys)
        if n < min_buys:
            continue
        acc = sum(1 for r in buys if r["actual_up"]) / n
        if acc > best_acc or (acc == best_acc and t > best_t):
            best_acc = acc
            best_t = t
            best_n = n

    if best_acc < 0:
        return 0.60, None, 0

    return best_t, float(best_acc), best_n


def walkforward_signal(up_p: float, buy_th: float, sell_th: float, ma20_rising: bool) -> str:
    """ウォークフォワード表示用: BUY / SELL / HOLD。"""
    if up_p >= buy_th:
        return "BUY"
    if up_p <= sell_th:
        if ma20_rising:
            return "HOLD"
        return "SELL"
    return "HOLD"


def live_signal(up_p: float, buy_th: float, sell_th: float, ma20_rising: bool) -> str:
    """judge 用: buy / sell / neutral。"""
    if up_p >= buy_th:
        return "buy"
    if up_p <= sell_th:
        if ma20_rising:
            return "neutral"
        return "sell"
    return "neutral"
