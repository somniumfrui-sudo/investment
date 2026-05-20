"""ウォークフォワード検証: ホライズン別（1/3/5営業日）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from calibration import calibrate_buy_threshold, walkforward_signal
from prediction_core import FEATURE_COLUMNS, build_features, make_model

HORIZONS = (1, 3, 5)


def _strip_tz(ts: pd.Timestamp) -> pd.Timestamp:
    if ts is None or not isinstance(ts, pd.Timestamp):
        return ts
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


def _proba_up(clf, x_row: np.ndarray) -> float:
    proba_row = clf.predict_proba(x_row.reshape(1, -1))[0]
    classes = list(clf.classes_)
    if 1 in classes:
        return float(proba_row[classes.index(1)])
    return float(proba_row.max())


def _run_walkforward_horizon(
    ticker: str,
    df: pd.DataFrame,
    raw: pd.DataFrame,
    horizon: int,
    train_months: int,
    test_months: int,
    total_months: int,
    sell_threshold: float,
) -> dict | None:
    feat = build_features(df, target_horizon=horizon)
    if feat.empty or len(feat) < 100:
        return None

    X_mat = feat[FEATURE_COLUMNS].values.astype(np.float64)
    mask = np.isfinite(X_mat).all(axis=1)
    feat = feat.loc[mask].copy()
    X_mat = feat[FEATURE_COLUMNS].values.astype(np.float64)
    y_vec = feat["target"].values.astype(int)

    idx = pd.DatetimeIndex([_strip_tz(pd.Timestamp(t)) for t in feat.index])
    feat = feat.copy()
    feat.index = idx

    close_series = raw["close"].astype(float).copy()
    close_series.index = pd.DatetimeIndex([_strip_tz(pd.Timestamp(t)) for t in close_series.index])
    close_s = close_series.reindex(feat.index)
    ma20 = close_s.rolling(20).mean()

    fut = close_s.shift(-horizon)
    next_pct = (fut / close_s - 1.0) * 100.0

    end = _strip_tz(pd.Timestamp(feat.index.max()))
    anchor = end - pd.DateOffset(months=total_months)
    anchor = _strip_tz(pd.Timestamp(anchor))
    first_idx = _strip_tz(pd.Timestamp(feat.index.min()))
    if anchor < first_idx:
        anchor = first_idx

    test_start = anchor + pd.DateOffset(months=train_months)
    test_start = _strip_tz(pd.Timestamp(test_start))

    rows: list[dict] = []

    while True:
        test_start_ts = _strip_tz(pd.Timestamp(test_start))
        if test_start_ts >= end:
            break
        test_end = _strip_tz(pd.Timestamp(test_start_ts + pd.DateOffset(months=test_months)))

        train_m = np.asarray((feat.index >= anchor) & (feat.index < test_start_ts), dtype=bool)
        test_m = np.asarray((feat.index >= test_start_ts) & (feat.index < test_end), dtype=bool)

        Xi_tr = X_mat[train_m]
        yi_tr = y_vec[train_m]

        if Xi_tr.shape[0] < 80 or len(np.unique(yi_tr)) < 2:
            test_start = test_end
            continue

        clf = make_model(light=False)
        clf.fit(Xi_tr, yi_tr)

        test_pos = np.flatnonzero(test_m)
        for j in test_pos:
            npc_raw = next_pct.iloc[j]
            if not np.isfinite(npc_raw):
                continue
            up_p = _proba_up(clf, X_mat[j])
            actual_up = bool(y_vec[j])
            npc = float(npc_raw)
            ma20_rising = False
            if j >= 5:
                ma20_rising = float(ma20.iloc[j]) > float(ma20.iloc[j - 5])

            rows.append(
                {
                    "date": feat.index[j],
                    "actual_up": actual_up,
                    "next_pct": npc,
                    "up_probability": up_p,
                    "ma20_rising": ma20_rising,
                }
            )

        test_start = test_end

    if not rows:
        return None

    buy_th, _, _ = calibrate_buy_threshold(rows)

    for r in rows:
        r["signal"] = walkforward_signal(
            float(r["up_probability"]),
            buy_th,
            sell_threshold,
            bool(r["ma20_rising"]),
        )

    def eval_rows(sub: list[dict]) -> tuple[int, int, float | None]:
        scored = [r for r in sub if r["signal"] in ("BUY", "SELL")]
        if not scored:
            return 0, 0, None
        ok = 0
        for r in scored:
            if r["signal"] == "BUY" and r["actual_up"]:
                ok += 1
            elif r["signal"] == "SELL" and (not r["actual_up"]):
                ok += 1
        return ok, len(scored), ok / len(scored)

    overall_ok, overall_n, overall_acc = eval_rows(rows)

    buy_rows = [r for r in rows if r["signal"] == "BUY"]
    sell_rows = [r for r in rows if r["signal"] == "SELL"]
    buy_ok = sum(1 for r in buy_rows if r["actual_up"])
    sell_ok = sum(1 for r in sell_rows if not r["actual_up"])
    buy_acc = (buy_ok / len(buy_rows)) if buy_rows else None
    sell_acc = (sell_ok / len(sell_rows)) if sell_rows else None

    monthly: dict[str, list[dict]] = {}
    for r in rows:
        if r["signal"] not in ("BUY", "SELL"):
            continue
        d = pd.Timestamp(r["date"])
        key = f"{d.year}-{d.month:02d}"
        monthly.setdefault(key, []).append(r)

    monthly_list: list[dict] = []
    for mkey in sorted(monthly.keys()):
        sub = monthly[mkey]
        ok, n, acc = eval_rows(sub)
        monthly_list.append({"month": mkey, "accuracy": acc, "n": n, "correct": ok})

    def format_row(r: dict) -> dict:
        d = pd.Timestamp(r["date"])
        ds = d.strftime("%Y-%m-%d")
        sig = r["signal"]
        npc = r["next_pct"]
        pct_s = f"{npc:+.2f}%" if npc == npc else "—"
        if sig == "HOLD":
            res = "➖ 対象外"
        elif sig == "BUY":
            res = "✅ 的中" if r["actual_up"] else "❌ 外れ"
        else:
            res = "✅ 的中" if not r["actual_up"] else "❌ 外れ"
        return {"日付": ds, "予測": sig, "実際": pct_s, "結果": res}

    hist_sorted = sorted(rows, key=lambda x: pd.Timestamp(x["date"]), reverse=True)
    history_table = [format_row(r) for r in hist_sorted[:10]]

    chart_df = pd.DataFrame(monthly_list)
    if not chart_df.empty:
        acc_num = pd.to_numeric(chart_df["accuracy"], errors="coerce")
        chart_df["accuracy_pct"] = (acc_num * 100.0).where(acc_num.notna())

    calibration_rows = [
        {
            "up_probability": r["up_probability"],
            "actual_up": r["actual_up"],
            "ma20_rising": r["ma20_rising"],
        }
        for r in rows
    ]

    lbl = {1: "翌日", 3: "3日後", 5: "5日後"}.get(horizon, f"{horizon}日後")

    return {
        "horizon": horizon,
        "label": lbl,
        "error": None,
        "overall_accuracy": overall_acc,
        "overall_correct": overall_ok,
        "overall_total": overall_n,
        "buy_accuracy": buy_acc,
        "buy_correct": buy_ok,
        "buy_total": len(buy_rows),
        "sell_accuracy": sell_acc,
        "sell_correct": sell_ok,
        "sell_total": len(sell_rows),
        "hold_count": sum(1 for r in rows if r["signal"] == "HOLD"),
        "monthly": chart_df,
        "history_table": history_table,
        "calibration_rows": calibration_rows,
        "optimal_buy_threshold_pct": round(buy_th * 100.0, 2),
        "params": {"train_months": train_months, "test_months": test_months, "total_months": total_months},
    }


def run_walkforward(
    ticker: str,
    train_months: int = 6,
    test_months: int = 1,
    total_months: int = 18,
    sell_threshold: float = 0.40,
) -> dict:
    df = yf.Ticker(ticker).history(period="2y", auto_adjust=True)
    if df is None or df.empty or len(df) < 80:
        return {"error": "価格データが不足しています（2年・80日以上が目安）。"}

    raw = df.rename(columns=str.lower)

    by_h: dict[int, dict] = {}
    errs: list[str] = []

    for h in HORIZONS:
        res = _run_walkforward_horizon(
            ticker, df, raw, h, train_months, test_months, total_months, sell_threshold
        )
        if res is None:
            errs.append(f"{h}営業日先の検証データが不足")
            continue
        by_h[h] = res

    if not by_h:
        return {"error": "ウォークフォワードを実行できませんでした。" + " ".join(errs)}

    comparison_rows = []
    for h in HORIZONS:
        if h not in by_h:
            comparison_rows.append(
                {
                    "期間": {1: "翌日", 3: "3日後", 5: "5日後"}.get(h, str(h)),
                    "BUY正解率": None,
                    "BUY件数": 0,
                    "最適BUY閾値": None,
                }
            )
            continue
        r = by_h[h]
        ba = r.get("buy_accuracy")
        comparison_rows.append(
            {
                "期間": r.get("label", str(h)),
                "BUY正解率": None if ba is None else round(ba * 100.0, 2),
                "BUY件数": r.get("buy_total", 0),
                "最適BUY閾値": r.get("optimal_buy_threshold_pct"),
            }
        )

    primary = by_h.get(1) or next(iter(by_h.values()))
    cal_by_h = {h: by_h[h]["calibration_rows"] for h in by_h}
    out = {
        "error": None if not errs else "一部ホライズンのみ: " + " ".join(errs),
        "by_horizon": by_h,
        "horizons": by_h,
        "calibration_rows_by_horizon": cal_by_h,
        "comparison_table": comparison_rows,
        # 後方互換: 従来キーは翌日を参照
        "overall_accuracy": primary.get("overall_accuracy"),
        "overall_correct": primary.get("overall_correct"),
        "overall_total": primary.get("overall_total"),
        "buy_accuracy": primary.get("buy_accuracy"),
        "buy_correct": primary.get("buy_correct"),
        "buy_total": primary.get("buy_total"),
        "sell_accuracy": primary.get("sell_accuracy"),
        "sell_correct": primary.get("sell_correct"),
        "sell_total": primary.get("sell_total"),
        "hold_count": primary.get("hold_count"),
        "monthly": primary.get("monthly"),
        "history_table": primary.get("history_table"),
        "calibration_rows": primary.get("calibration_rows"),
        "optimal_buy_threshold_pct": primary.get("optimal_buy_threshold_pct"),
        "params": primary.get("params"),
    }
    return out
