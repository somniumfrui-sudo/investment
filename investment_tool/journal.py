"""トレード日誌: journal.csv の保存・照合・集計・再学習用エクスポート。"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from prediction_core import FEATURE_COLUMNS, build_features

JOURNAL_PATH = Path(__file__).resolve().parent / "journal.csv"

COLUMNS = [
    "date",
    "ticker",
    "signal",
    "period",
    "ai_prob",
    "chart_signal",
    "sentiment",
    "total_score",
    "entry_price",
    "exit_price",
    "actual_return",
    "pnl",
    "hit",
    "budget",
]


def _period_to_offset(period: str) -> int | None:
    m = {"翌日": 1, "3日後": 3, "5日後": 5}
    return m.get(str(period).strip())


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def _load() -> pd.DataFrame:
    if not JOURNAL_PATH.exists():
        return _empty_df()
    try:
        df = pd.read_csv(JOURNAL_PATH)
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = np.nan
        return df[COLUMNS]
    except Exception:
        return _empty_df()


def _save_df(df: pd.DataFrame) -> None:
    df[COLUMNS].to_csv(JOURNAL_PATH, index=False, encoding="utf-8-sig")


def save_signal(
    signal_date: date | str,
    ticker: str,
    signal: str,
    period: str,
    ai_prob: float,
    chart_signal: str,
    sentiment: str,
    total_score: float,
    entry_price: float,
    budget: float,
) -> None:
    """スキャン時にシグナルを CSV に追記。"""
    df = _load()
    row = {
        "date": str(signal_date) if not isinstance(signal_date, str) else signal_date,
        "ticker": str(ticker).upper().strip(),
        "signal": str(signal).upper().strip(),
        "period": str(period).strip(),
        "ai_prob": round(float(ai_prob), 4),
        "chart_signal": str(chart_signal or "").lower(),
        "sentiment": str(sentiment or ""),
        "total_score": round(float(total_score), 6),
        "entry_price": round(float(entry_price), 6),
        "exit_price": np.nan,
        "actual_return": np.nan,
        "pnl": np.nan,
        "hit": -1,
        "budget": round(float(budget), 2),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save_df(df)


def save_top3_buy_from_scan(
    scan_rows: list[dict],
    budget: float,
    signal_date: date | None = None,
) -> int:
    """
    ランキング上位から最大3銘柄を対象に、BUY になっている期間だけ日誌に保存。
    戻り値: 追記した行数。
    """
    sd = signal_date or date.today()
    n_saved = 0
    for r in scan_rows[:3]:
        t = r.get("ticker")
        if not t:
            continue
        entry = _fetch_close_on_or_before(t, sd)
        if entry is None or entry <= 0:
            continue
        score = float(r.get("score", 0.0))
        chart_sig = str(r.get("chart_signal") or "neutral")
        sent = str(r.get("sent_signal") or "neutral")

        checks = [
            ("翌日", r.get("pred_1d"), r.get("prob_1d")),
            ("3日後", r.get("pred_3d"), r.get("prob_3d")),
            ("5日後", r.get("pred_5d"), r.get("prob_5d")),
        ]
        for period, pred, prob in checks:
            if pred is None or prob is None:
                continue
            pred_s = str(pred).upper()
            if "BUY" not in pred_s:
                continue
            save_signal(
                sd,
                t,
                "BUY",
                period,
                float(prob) * 100.0,
                chart_sig,
                sent,
                score,
                entry,
                budget,
            )
            n_saved += 1
    return n_saved


def _fetch_close_on_or_before(ticker: str, d: date) -> float | None:
    try:
        start = pd.Timestamp(d) - pd.Timedelta(days=7)
        end = pd.Timestamp(d) + pd.Timedelta(days=2)
        df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if df is None or df.empty:
            return None
        idx = pd.DatetimeIndex(
            [x.tz_localize(None).normalize() if x.tzinfo else x.normalize() for x in df.index]
        )
        df = df.copy()
        df.index = idx
        target = pd.Timestamp(d).normalize()
        sub = df[df.index <= target]
        if sub.empty:
            return None
        return float(sub["Close"].iloc[-1])
    except Exception:
        return None


def _exit_close_after_n_sessions(ticker: str, signal_date: date, n_sessions: int) -> float | None:
    """シグナル日の終値を起点に、n_sessions 営業日後の終値（yfinance の取引カレンダー）。"""
    if n_sessions < 1:
        return None
    try:
        start = pd.Timestamp(signal_date) - pd.Timedelta(days=5)
        end = pd.Timestamp.now() + pd.Timedelta(days=5)
        df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if df is None or df.empty:
            return None
        idx = pd.DatetimeIndex(
            [x.tz_localize(None).normalize() if x.tzinfo else x.normalize() for x in df.index]
        )
        df = df.sort_index()
        df.index = idx
        d0 = pd.Timestamp(signal_date).normalize()
        pos = None
        for i, dt in enumerate(df.index):
            if dt >= d0:
                pos = i
                break
        if pos is None:
            return None
        j = pos + n_sessions
        if j >= len(df):
            return None
        return float(df["Close"].iloc[j])
    except Exception:
        return None


def update_results() -> int:
    """
    hit=-1 の行について、経過営業日が足りれば exit を取得し actual_return / pnl / hit を更新。
    戻り値: 更新した行数。
    """
    df = _load()
    if df.empty:
        return 0
    updated = 0
    for i, row in df.iterrows():
        if int(row.get("hit", -1)) != -1:
            continue
        ticker = str(row["ticker"])
        period = str(row["period"])
        n = _period_to_offset(period)
        if n is None:
            continue
        try:
            sd = pd.Timestamp(str(row["date"])).date()
        except Exception:
            continue
        exit_p = _exit_close_after_n_sessions(ticker, sd, n)
        if exit_p is None or not np.isfinite(exit_p):
            continue
        entry = float(row["entry_price"])
        if entry <= 0:
            continue
        ret_pct = (exit_p - entry) / entry * 100.0
        sig = str(row["signal"]).upper()
        budget = float(row["budget"])

        if sig == "BUY":
            pnl = budget * (ret_pct / 100.0)
            hit = 1 if ret_pct > 0 else 0
        elif sig == "SELL":
            ret_pct_short = -ret_pct
            pnl = budget * (ret_pct_short / 100.0)
            hit = 1 if ret_pct < 0 else 0
        else:
            pnl = 0.0
            hit = 0

        df.at[i, "exit_price"] = round(exit_p, 6)
        df.at[i, "actual_return"] = round(ret_pct, 6)
        df.at[i, "pnl"] = round(pnl, 4)
        df.at[i, "hit"] = int(hit)
        updated += 1

    if updated:
        _save_df(df)
    return updated


def get_stats() -> dict:
    """集計結果を辞書で返す。"""
    df = _load()
    if df.empty:
        return {"empty": True}

    resolved = df[df["hit"].isin([0, 1])].copy()
    pending = int((df["hit"] == -1).sum())

    def hit_rate(sub: pd.DataFrame) -> float | None:
        if sub.empty:
            return None
        return float(sub["hit"].mean())

    overall = hit_rate(resolved)

    by_ticker = {}
    for t, g in resolved.groupby("ticker"):
        hr = hit_rate(g)
        if hr is not None:
            by_ticker[t] = {"rate": hr, "n": len(g)}

    best_ticker = None
    if by_ticker:
        best_ticker = max(by_ticker.items(), key=lambda x: (x[1]["rate"], x[1]["n"]))

    by_period = {}
    for p, g in resolved.groupby("period"):
        hr = hit_rate(g)
        if hr is not None:
            by_period[p] = {"rate": hr, "n": len(g)}

    best_period = None
    if by_period:
        best_period = max(by_period.items(), key=lambda x: (x[1]["rate"], x[1]["n"]))

    prob_buckets = [
        ("60-65%", 60, 65),
        ("65-70%", 65, 70),
        ("70%以上", 70, None),
    ]
    bucket_stats = []
    for label, lo, hi in prob_buckets:
        if hi is None:
            sub = resolved[resolved["ai_prob"] >= lo]
        else:
            sub = resolved[(resolved["ai_prob"] >= lo) & (resolved["ai_prob"] < hi)]
        hr = hit_rate(sub)
        bucket_stats.append({"bucket": label, "rate": hr, "n": len(sub)})

    resolved_m = resolved.copy()
    resolved_m["month"] = pd.to_datetime(resolved_m["date"], errors="coerce").dt.to_period("M").astype(str)
    monthly_hit = []
    for m, g in resolved_m.groupby("month"):
        hr = hit_rate(g)
        if m and str(m) != "NaT":
            monthly_hit.append({"month": m, "rate": hr, "n": len(g)})

    pnl_df = df[df["hit"].isin([0, 1])].copy()
    pnl_df["month"] = pd.to_datetime(pnl_df["date"], errors="coerce").dt.to_period("M").astype(str)
    monthly_pnl = pnl_df.groupby("month")["pnl"].sum().reset_index()
    monthly_pnl.columns = ["month", "pnl"]

    total_pnl = float(resolved["pnl"].sum()) if "pnl" in resolved.columns and not resolved.empty else 0.0

    return {
        "empty": False,
        "overall_hit_rate": overall,
        "resolved_n": len(resolved),
        "pending_n": pending,
        "best_ticker": best_ticker,
        "by_period": by_period,
        "best_period": best_period,
        "bucket_stats": bucket_stats,
        "monthly_hit": monthly_hit,
        "monthly_pnl": monthly_pnl,
        "total_pnl": total_pnl,
        "raw": df,
    }


def export_for_retraining() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    的中が確定した行から、各ホライズンごとに (X, y) を構築。
    戻り値: {1: (X, y), 3: (X, y), 5: (X, y)}（データが無いホライズンはキーなし）
    """
    df = _load()
    resolved = df[df["hit"].isin([0, 1])].copy()
    by_h: dict[int, list[tuple[np.ndarray, int]]] = {1: [], 3: [], 5: []}

    for _, row in resolved.iterrows():
        period = str(row["period"])
        h = _period_to_offset(period)
        if h is None:
            continue
        try:
            sd = pd.Timestamp(str(row["date"])).date()
        except Exception:
            continue
        ticker = str(row["ticker"])
        pair = _features_label_at_date(ticker, sd, h)
        if pair is None:
            continue
        Xrow, y = pair
        by_h[h].append((Xrow, y))

    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for h, lst in by_h.items():
        if not lst:
            continue
        Xs = np.stack([x[0] for x in lst], axis=0)
        ys = np.array([x[1] for x in lst], dtype=int)
        if len(np.unique(ys)) < 2:
            continue
        out[h] = (Xs, ys)
    return out


def _features_label_at_date(ticker: str, signal_date: date, horizon: int) -> tuple[np.ndarray, int] | None:
    """シグナル日時点の特徴量行と、ホライズン先の上昇ラベル。"""
    try:
        end = pd.Timestamp(signal_date) + pd.Timedelta(days=1)
        start = pd.Timestamp(signal_date) - pd.Timedelta(days=500)
        hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        if hist is None or len(hist) < 80:
            return None
        feat = build_features(hist, target_horizon=horizon)
        if feat.empty:
            return None
        idx = feat.index
        norm_idx = pd.DatetimeIndex(
            [pd.Timestamp(x).tz_localize(None).normalize() if pd.Timestamp(x).tzinfo else pd.Timestamp(x).normalize() for x in idx]
        )
        feat = feat.copy()
        feat.index = norm_idx
        sd = pd.Timestamp(signal_date).normalize()
        if sd not in feat.index:
            sub = feat[feat.index <= sd]
            if sub.empty:
                return None
            r = sub.iloc[-1]
        else:
            r = feat.loc[sd]
        X = r[FEATURE_COLUMNS].values.astype(np.float64)
        if not np.isfinite(X).all():
            return None
        y = int(r["target"])
        return X, y
    except Exception:
        return None


def confirmed_sample_count() -> int:
    df = _load()
    return int(df["hit"].isin([0, 1]).sum())
