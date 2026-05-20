"""複数ホライズン（1/3/5営業日）の騰落予測。"""

from __future__ import annotations

import copy
import logging
import threading
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from calibration import calibrate_buy_threshold, live_signal, ma20_is_uptrend
from prediction_core import FEATURE_COLUMNS, build_features, make_model

HOLDOUT_DAYS = 60
HORIZONS = (1, 3, 5)
JOURNAL_BOOST_PATH = Path(__file__).resolve().parent / "journal_boost.npz"
MIN_JOURNAL_RETRAIN_SAMPLES = 20

_PRED_CACHE_LOCK = threading.Lock()

logger = logging.getLogger(__name__)
if not logging.root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [predictor] %(message)s",
    )


def _positive_up_probability(clf, X_row: np.ndarray) -> float:
    """classes_ に依存せず「上昇=1」の確率列を取得する。"""
    proba_row = clf.predict_proba(X_row.reshape(1, -1))[0]
    classes = np.asarray(clf.classes_)
    pos_idx = None
    for i, c in enumerate(classes):
        if int(c) == 1:
            pos_idx = i
            break
    if pos_idx is not None:
        return float(proba_row[pos_idx])
    if len(proba_row) >= 2:
        return float(proba_row[-1])
    return float(np.clip(proba_row.max(), 0.0, 1.0))


def _load_journal_extra(horizon: int) -> tuple[np.ndarray, np.ndarray] | None:
    if not JOURNAL_BOOST_PATH.exists():
        return None
    try:
        z = np.load(JOURNAL_BOOST_PATH, allow_pickle=False)
        xk, yk = f"h{horizon}_X", f"h{horizon}_y"
        if xk not in z.files or yk not in z.files:
            return None
        Xe, ye = z[xk], z[yk]
        if Xe.ndim != 2 or len(ye) != len(Xe):
            return None
        return Xe.astype(np.float64), ye.astype(int)
    except Exception as e:
        logger.warning("journal_boost の読み込みに失敗: %s", e)
        return None


def retrain_from_journal() -> tuple[int, str]:
    """journal.csv の確定行から特徴量を復元し、journal_boost.npz に保存して予測に反映。"""
    from journal import export_for_retraining, confirmed_sample_count

    n_conf = confirmed_sample_count()
    if n_conf < MIN_JOURNAL_RETRAIN_SAMPLES:
        return 0, f"まだデータが少ないです（{n_conf}/{MIN_JOURNAL_RETRAIN_SAMPLES}件）"

    ex = export_for_retraining()
    total = sum(v[0].shape[0] for v in ex.values())
    if total < MIN_JOURNAL_RETRAIN_SAMPLES:
        return 0, f"特徴量を復元できた件数が不足です（{total}/{MIN_JOURNAL_RETRAIN_SAMPLES}件）"

    kw: dict[str, np.ndarray] = {}
    for h, (Xa, ya) in ex.items():
        kw[f"h{h}_X"] = Xa
        kw[f"h{h}_y"] = ya
    np.savez_compressed(JOURNAL_BOOST_PATH, **kw)
    return total, f"再学習完了。追加データ: {total}件"


def _time_series_cv_metrics(
    X: np.ndarray, y: np.ndarray, n_splits: int = 5, *, light: bool = False
) -> tuple[float | None, float | None]:
    if len(X) < 80:
        return None, None
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs: list[float] = []
    aucs: list[float] = []
    for train_idx, test_idx in tscv.split(X):
        if len(test_idx) < 5 or len(train_idx) < 40:
            continue
        y_tr, y_te = y[train_idx], y[test_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        clf = make_model(light=light)
        clf.fit(X[train_idx], y_tr)
        pred = clf.predict(X[test_idx])
        accs.append(accuracy_score(y_te, pred))
        proba_full = clf.predict_proba(X[test_idx])
        classes = np.asarray(clf.classes_)
        if proba_full.shape[1] >= 2 and np.isin(1, classes):
            j = int(np.flatnonzero(classes == 1)[0])
            proba_pos = np.clip(proba_full[:, j], 1e-7, 1.0 - 1e-7)
            aucs.append(roc_auc_score(y_te, proba_pos))
    if not accs:
        return None, None
    return float(np.mean(accs)), (float(np.mean(aucs)) if aucs else None)


def _predict_one_horizon(
    horizon: int,
    df: pd.DataFrame,
    calibration_rows: list[dict] | None,
    sell_threshold: float,
    ma_up: bool,
    *,
    skip_time_series_cv: bool = False,
    light_models: bool = False,
) -> dict | None:
    try:
        feat = build_features(df, target_horizon=horizon)
        if len(feat) < 100:
            logger.info(
                "horizon=%s: 特徴量行が不足 (%s行・100未満)",
                horizon,
                len(feat),
            )
            return None

        X_df = feat[FEATURE_COLUMNS]
        X = X_df.values.astype(np.float64)
        y = feat["target"].values.astype(int)

        mask = np.isfinite(X).all(axis=1)
        X, y = X[mask], y[mask]
        if len(X) < 100:
            logger.info(
                "horizon=%s: 有限行が不足 (%s行)",
                horizon,
                len(X),
            )
            return None

        n = len(X)
        if n < HOLDOUT_DAYS + 90:
            logger.info(
                "horizon=%s: 全行数がホールドアウト要件未満 (n=%s < %s)",
                horizon,
                n,
                HOLDOUT_DAYS + 90,
            )
            return None

        split = n - HOLDOUT_DAYS
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        if len(np.unique(y_train)) < 2:
            logger.info("horizon=%s: 学習ラベルが単一クラスのみ", horizon)
            return None

        if skip_time_series_cv:
            cv_acc, cv_auc = None, None
        else:
            cv_acc, cv_auc = _time_series_cv_metrics(X_train, y_train, n_splits=5, light=light_models)

        journal_pack = _load_journal_extra(horizon)
        Xe: np.ndarray | None = None
        ye: np.ndarray | None = None
        if journal_pack is not None:
            Xe, ye = journal_pack

        if Xe is not None and ye is not None and Xe.shape[1] == X_train.shape[1]:
            X_train_j = np.vstack([X_train, Xe])
            y_train_j = np.concatenate([y_train, ye])
        else:
            X_train_j, y_train_j = X_train, y_train

        holdout = make_model(light=light_models)
        holdout.fit(X_train_j, y_train_j)
        val_acc = (
            float(holdout.score(X_val, y_val)) if len(np.unique(y_val)) > 1 else float("nan")
        )

        final_clf = make_model(light=light_models)
        if Xe is not None and ye is not None and Xe.shape[1] == X.shape[1]:
            X_final = np.vstack([X[:-1], Xe])
            y_final = np.concatenate([y[:-1], ye])
        else:
            X_final, y_final = X[:-1], y[:-1]
        final_clf.fit(X_final, y_final)

        up_p = _positive_up_probability(final_clf, X[-1])

        if calibration_rows:
            buy_th, cal_wf_acc, cal_wf_n = calibrate_buy_threshold(calibration_rows)
            cal_source = "walkforward"
        else:
            buy_th, cal_wf_acc, cal_wf_n = 0.60, None, 0
            cal_source = "default"

        signal = live_signal(up_p, buy_th, sell_threshold, ma_up)
        would_sell = up_p <= sell_threshold
        sell_suppressed = bool(would_sell and ma_up)

        label = {1: "翌日", 3: "3日後", 5: "5日後"}.get(horizon, f"{horizon}日後")

        acc_txt = f"{val_acc*100:.1f}%" if val_acc == val_acc else "—"
        cv_txt = f"{cv_acc*100:.1f}%" if cv_acc is not None else "—"
        auc_txt = f"{cv_auc*100:.1f}%" if cv_auc is not None else "—"
        reason = (
            f"{label}上昇確率 {up_p*100:.1f}% ・ BUY閾値 {buy_th*100:.1f}%（{cal_source}）・ "
            f"ホールドアウト末尾{HOLDOUT_DAYS}日 正解率 {acc_txt} / CV {cv_txt} / AUC {auc_txt}"
        )

        sig_ui = signal.upper() if signal in ("buy", "sell") else "HOLD"

        return {
            "horizon": horizon,
            "label": label,
            "signal": signal,
            "signal_display": sig_ui,
            "up_probability": up_p,
            "val_accuracy": val_acc,
            "cv_accuracy_mean": cv_acc,
            "cv_auc_mean": cv_auc,
            "reason": reason,
            "optimal_buy_threshold_pct": round(buy_th * 100.0, 2),
            "buy_threshold_source": cal_source,
            "calibration_walkforward_buy_accuracy": cal_wf_acc,
            "calibration_walkforward_buy_n": cal_wf_n,
            "sell_suppressed_by_trend": sell_suppressed,
            "holdout_days": HOLDOUT_DAYS,
        }
    except Exception as e:
        logger.error(
            "horizon=%s の予測で例外: %s\n%s",
            horizon,
            e,
            traceback.format_exc(),
        )
        return None


def predict_direction(
    ticker: str,
    calibration_rows_by_horizon: dict[int, list[dict]] | None = None,
    sell_threshold: float = 0.40,
    *,
    history_period: str = "2y",
    skip_time_series_cv: bool = False,
    use_light_models: bool = False,
    prediction_cache: dict[str, dict] | None = None,
    cache_date: str | None = None,
) -> dict:
    """
    1 / 3 / 5 営業日先の上昇をそれぞれ別モデルで予測。
    calibration_rows_by_horizon: ウォークフォワード由来のキャリブレーション（単一銘柄分析用）。
    prediction_cache + cache_date: 同一営業日内の再スキャンで再学習を避ける（キーは銘柄×設定）。
    """
    if calibration_rows_by_horizon is None:
        calibration_rows_by_horizon = {}

    use_day_cache = (
        prediction_cache is not None
        and cache_date
        and len(calibration_rows_by_horizon) == 0
    )
    cache_key = None
    if use_day_cache:
        cache_key = (
            f"{cache_date}|{ticker.upper()}|{history_period}|"
            f"cv{int(not skip_time_series_cv)}|lt{int(use_light_models)}"
        )
        with _PRED_CACHE_LOCK:
            if cache_key in prediction_cache:
                logger.info("%s: 当日キャッシュヒット %s", ticker, cache_key)
                return copy.deepcopy(prediction_cache[cache_key])

    t = yf.Ticker(ticker)
    df = t.history(period=history_period, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.Index(df.columns.get_level_values(0))

    min_rows = 100 if history_period in ("1y", "6mo") else 120
    if df is None or df.empty or len(df) < min_rows:
        logger.warning("%s: 履歴が不足 rows=%s period=%s", ticker, 0 if df is None else len(df), history_period)
        return {
            "error": f"{history_period} のデータが不足しています（目安 {min_rows} 営業日以上）。",
            "signal": "neutral",
            "horizons": {},
        }

    close_live = df["Close"].astype(float)
    ma_up = ma20_is_uptrend(close_live)

    horizons_out: dict[int, dict] = {}
    errors: list[str] = []

    for h in HORIZONS:
        cal_rows = calibration_rows_by_horizon.get(h)
        one = _predict_one_horizon(
            h,
            df,
            cal_rows,
            sell_threshold,
            ma_up,
            skip_time_series_cv=skip_time_series_cv,
            light_models=use_light_models,
        )
        if one is None:
            errors.append(f"{h}営業日先の学習に失敗（データ不足など）")
            logger.info("%s: horizon=%s をスキップ（理由は上記ログ参照）", ticker, h)
            continue
        horizons_out[h] = one

    if not horizons_out:
        return {
            "error": "全ホライズンで予測できませんでした。" + " ".join(errors),
            "signal": "neutral",
            "horizons": {},
        }

    h3 = horizons_out.get(3)
    h5 = horizons_out.get(5)
    sig3 = h3["signal"] if h3 else None
    sig5 = h5["signal"] if h5 else None

    if sig3 == "buy" and sig5 == "buy":
        composite = "buy"
    elif sig3 == "sell" and sig5 == "sell":
        composite = "sell"
    else:
        composite = "neutral"

    jp_names = [
        "MA5比",
        "MA20比",
        "MA50比",
        "MA5-20乖離",
        "RSI",
        "RSI(3日差分)",
        "MACD",
        "MACDヒスト",
        "出来高変化率",
        "出来高/20日MA",
        "前日騰落率",
        "5日騰落率",
        "10日ボラ",
        "BB %b",
        "曜日",
    ]

    err_msg = None
    if errors:
        err_msg = " ".join(errors)

    logger.info(
        "%s: 予測完了 horizons=%s probs=%s err=%s",
        ticker,
        list(horizons_out.keys()),
        {h: round(horizons_out[h]["up_probability"], 4) for h in horizons_out},
        err_msg,
    )

    result = {
        "error": err_msg,
        "signal": composite,
        "horizons": horizons_out,
        "ma20_uptrend": ma_up,
        "feature_names": jp_names,
        "model_note": "RandomForest + HistGradientBoosting（ソフト投票）× ホライズン別",
        "both_long_horizons_buy": bool(sig3 == "buy" and sig5 == "buy"),
    }

    if use_day_cache and cache_key:
        with _PRED_CACHE_LOCK:
            prediction_cache[cache_key] = copy.deepcopy(result)

    return result
