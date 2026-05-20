"""RSI / MACD / ボリンジャー（pandas のみ・pandas-ta 非依存）。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder 型 RSI（一般的なテクニカル指標と同系）。"""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0))
    avg_g = gain.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    avg_l = loss.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    rs = avg_g / avg_l.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out


def macd_line_signal(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD ラインとシグナル（EMA ベース）。"""
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig}, index=close.index)


def bollinger_bands(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(length).mean()
    sd = close.rolling(length).std()
    lower = mid - std * sd
    upper = mid + std * sd
    return pd.DataFrame({"bb_lower": lower, "bb_mid": mid, "bb_upper": upper}, index=close.index)
