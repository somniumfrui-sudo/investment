"""チャート分析: RSI / MACD / ボリンジャーバンドの多数決。"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from indicators import bollinger_bands, macd_line_signal, rsi_wilder


def _rsi_signal(val: float | None) -> str:
    if val is None or pd.isna(val):
        return "neutral"
    if val > 70:
        return "sell"
    if val < 30:
        return "buy"
    return "neutral"


def _macd_signal(macd: float, signal: float, prev_macd: float, prev_signal: float) -> str:
    if any(pd.isna(x) for x in (macd, signal, prev_macd, prev_signal)):
        return "neutral"
    # ゴールデンクロス: MACDがシグナルを下から上抜け
    if prev_macd <= prev_signal and macd > signal:
        return "buy"
    # デッドクロス: MACDがシグナルを上から下抜け
    if prev_macd >= prev_signal and macd < signal:
        return "sell"
    return "neutral"


def _bb_signal(close: float, lower: float, upper: float) -> str:
    if any(pd.isna(x) for x in (close, lower, upper)) or upper <= lower:
        return "neutral"
    width = upper - lower
    if width <= 0:
        return "neutral"
    pos = (close - lower) / width  # 0=下限付近, 1=上限付近
    if pos <= 0.15:
        return "buy"
    if pos >= 0.85:
        return "sell"
    return "neutral"


def analyze_chart(ticker: str) -> dict:
    """
    過去90日のデータでテクニカル分析。
    戻り値: signal (buy/neutral/sell), strength (-1..1), details
    """
    t = yf.Ticker(ticker)
    df = t.history(period="90d", auto_adjust=True)
    if df is None or df.empty or len(df) < 30:
        return {
            "signal": "neutral",
            "strength": 0.0,
            "error": "データが不足しています（90日分が必要です）。",
            "details": {},
        }

    ohlcv = df.rename(columns=str.lower)
    c = ohlcv["close"]
    ohlcv["rsi_14"] = rsi_wilder(c, 14)
    macd_df = macd_line_signal(c, 12, 26, 9)
    ohlcv["macd_line"] = macd_df["macd"]
    ohlcv["macd_sig"] = macd_df["macd_signal"]
    bb = bollinger_bands(c, 20, 2.0)
    ohlcv["bb_lower"] = bb["bb_lower"]
    ohlcv["bb_upper"] = bb["bb_upper"]

    last = ohlcv.iloc[-1]
    prev = ohlcv.iloc[-2]

    rsi_val = float(last["rsi_14"]) if not pd.isna(last["rsi_14"]) else None
    macd_now = float(last["macd_line"])
    sig_now = float(last["macd_sig"])
    macd_prev = float(prev["macd_line"])
    sig_prev = float(prev["macd_sig"])

    close = float(last["close"])
    lower = float(last["bb_lower"])
    upper = float(last["bb_upper"])

    s_rsi = _rsi_signal(rsi_val)
    s_macd = _macd_signal(macd_now, sig_now, macd_prev, sig_prev)
    s_bb = _bb_signal(close, lower, upper)

    votes = {"rsi": s_rsi, "macd": s_macd, "bollinger": s_bb}
    buy_n = sum(1 for v in votes.values() if v == "buy")
    sell_n = sum(1 for v in votes.values() if v == "sell")

    if buy_n >= 2:
        signal = "buy"
    elif sell_n >= 2:
        signal = "sell"
    else:
        signal = "neutral"

    strength = (buy_n - sell_n) / 3.0

    reasons = []
    reasons.append(f"RSI(14)={rsi_val:.2f} → {s_rsi}（70超=売り・30未満=買い）")
    reasons.append(
        f"MACD vs シグナル: 現値 {macd_now:.4f}/{sig_now:.4f} → {s_macd}（クロス判定）"
    )
    reasons.append(
        f"BB帯内位置: 終値${close:.2f}（下限~上限） → {s_bb}（帯の下15%/上15%）"
    )

    return {
        "signal": signal,
        "strength": float(max(-1.0, min(1.0, strength))),
        "details": {
            "rsi": rsi_val,
            "macd": macd_now,
            "macd_signal": sig_now,
            "bb_lower": lower,
            "bb_upper": upper,
            "close": close,
            "votes": votes,
            "reasons": reasons,
        },
    }
