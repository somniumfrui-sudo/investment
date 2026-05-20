"""AI予測・ウォークフォワード共通: 特徴量列・特徴量生成・アンサンブルモデル。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier

from indicators import bollinger_bands, macd_line_signal, rsi_wilder

FEATURE_COLUMNS: list[str] = [
    "ma5_ratio",
    "ma20_ratio",
    "ma50_ratio",
    "ma_spread",
    "rsi",
    "rsi_delta3",
    "macd",
    "macd_hist",
    "vol_chg",
    "vol_rel_ma20",
    "ret_1",
    "ret_5",
    "volatility_10",
    "bb_pctb",
    "dow",
]


def build_features(df: pd.DataFrame, *, target_horizon: int = 1) -> pd.DataFrame:
    o = df.rename(columns=str.lower).copy()
    c = o["close"]
    v = o["volume"].replace(0, np.nan)

    o["ma5"] = c.rolling(5).mean()
    o["ma20"] = c.rolling(20).mean()
    o["ma50"] = c.rolling(50).mean()
    o["rsi_14"] = rsi_wilder(c, 14)
    m = macd_line_signal(c, 12, 26, 9)
    o["macd_line"] = m["macd"]
    o["macd_sig"] = m["macd_signal"]
    bb = bollinger_bands(c, 20, 2.0)
    bw = (bb["bb_upper"] - bb["bb_lower"]).replace(0, np.nan)
    pct_b = (c - bb["bb_lower"]) / bw

    vol_ma20 = v.rolling(20).mean()

    idx = o.index
    if hasattr(idx, "dayofweek"):
        dow = pd.Series(idx.dayofweek, index=idx, dtype=float)
    else:
        dow = pd.Series(0.0, index=idx)

    feat = pd.DataFrame(
        {
            "ma5_ratio": o["ma5"] / c - 1.0,
            "ma20_ratio": o["ma20"] / c - 1.0,
            "ma50_ratio": o["ma50"] / c - 1.0,
            "ma_spread": o["ma5"] / o["ma20"].replace(0, np.nan) - 1.0,
            "rsi": o["rsi_14"],
            "rsi_delta3": o["rsi_14"] - o["rsi_14"].shift(3),
            "macd": o["macd_line"],
            "macd_hist": o["macd_line"] - o["macd_sig"],
            "vol_chg": v.pct_change(),
            "vol_rel_ma20": v / vol_ma20.replace(0, np.nan) - 1.0,
            "ret_1": c.pct_change(),
            "ret_5": c.pct_change(5),
            "volatility_10": c.pct_change().rolling(10).std(),
            "bb_pctb": pct_b.clip(0, 1),
            "dow": dow,
        },
        index=o.index,
    )
    h = max(1, int(target_horizon))
    feat["target"] = (c.shift(-h) > c).astype(float)
    return feat.dropna()


def make_model(*, light: bool = False) -> VotingClassifier:
    if light:
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        hgb = HistGradientBoostingClassifier(
            max_depth=8,
            max_iter=120,
            learning_rate=0.08,
            min_samples_leaf=25,
            l2_regularization=0.1,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=15,
        )
    else:
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=14,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        hgb = HistGradientBoostingClassifier(
            max_depth=10,
            max_iter=300,
            learning_rate=0.05,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
        )
    return VotingClassifier(
        estimators=[("rf", rf), ("hgb", hgb)],
        voting="soft",
        n_jobs=-1,
    )
