"""複数銘柄スキャン・ランキング。"""

from __future__ import annotations

import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

from analyzer import analyze_chart

logger = logging.getLogger(__name__)
if not logging.root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [scanner] %(message)s",
    )

from predictor import predict_direction
from sentiment import analyze_sentiment

DEFAULT_TICKERS: tuple[str, ...] = (
    "PLTR",
    "NVDA",
    "AMD",
    "SOXL",
    "SQ",
    "MSTR",
    "TSM",
    "CRWD",
    "COIN",
    "HOOD",
)


def _signal_component(sig: str) -> float:
    s = (sig or "").lower()
    if s == "buy":
        return 1.0
    if s == "sell":
        return -1.0
    return 0.0


def _jp_sentiment(sig: str) -> str:
    m = {"buy": "ポジ", "sell": "ネガ", "neutral": "中立"}
    return m.get((sig or "").lower(), "—")


def _jp_chart(sig: str) -> str:
    m = {"buy": "買い", "sell": "売り", "neutral": "中立"}
    return m.get((sig or "").lower(), "—")


def _scan_one_ticker(
    t: str,
    prediction_cache: dict[str, dict],
    cache_date: str,
) -> tuple[str, dict]:
    """1銘柄分（チャート・感情・AI）。スレッドから呼ばれる。"""
    try:
        chart = analyze_chart(t)
    except Exception:
        chart = {"signal": "neutral", "strength": 0.0}
    try:
        sent = analyze_sentiment(t)
    except Exception:
        sent = {"signal": "neutral"}
    try:
        ai = predict_direction(
            t,
            calibration_rows_by_horizon=None,
            history_period="1y",
            skip_time_series_cv=True,
            use_light_models=True,
            prediction_cache=prediction_cache,
            cache_date=cache_date,
        )
        hz = ai.get("horizons") or {}
        if ai.get("error"):
            logger.warning("%s: AI predict メッセージ: %s", t, ai["error"])
        if not hz:
            logger.error("%s: horizons が空です。", t)
        else:
            probs = [hz.get(h, {}).get("up_probability") for h in (1, 3, 5)]
            logger.info("%s: AI up_probability (1d/3d/5d) = %s", t, probs)
    except Exception:
        logger.error(
            "%s: predict_direction で例外\n%s",
            t,
            traceback.format_exc(),
        )
        ai = {"error": "予測失敗", "horizons": {}}

    chart_score = float(chart.get("strength", 0.0))
    sent_score = _signal_component(sent.get("signal", "neutral"))

    h = ai.get("horizons") or {}
    h1 = h.get(1, {})
    h3 = h.get(3, {})
    h5 = h.get(5, {})
    p1 = float(h1.get("up_probability") or 0.0)
    p3 = float(h3.get("up_probability") or 0.0)
    p5 = float(h5.get("up_probability") or 0.0)

    score = chart_score * 1.0 + sent_score * 1.0 + p3 * 1.5 + p5 * 1.5

    sig1 = h1.get("signal_display") or "—"
    sig3 = h3.get("signal_display") or "—"
    sig5 = h5.get("signal_display") or "—"

    row = {
        "ticker": t,
        "score": score,
        "prob_1d": p1,
        "prob_3d": p3,
        "prob_5d": p5,
        "pred_1d": sig1,
        "pred_3d": sig3,
        "pred_5d": sig5,
        "chart_jp": _jp_chart(chart.get("signal", "neutral")),
        "sent_jp": _jp_sentiment(sent.get("signal", "neutral")),
        "chart_signal": chart.get("signal"),
        "sent_signal": sent.get("signal"),
        "ai_error": ai.get("error"),
    }
    return t, row


def scan_tickers(
    tickers: list[str] | None = None,
    *,
    prediction_cache: dict[str, dict] | None = None,
    cache_date: str | None = None,
    max_workers: int = 4,
) -> tuple[list[dict], float]:
    """
    各銘柄を並列分析。戻り値: (ランキング行, 経過秒)。
    prediction_cache / cache_date を渡すと当日の predict_direction 結果を再利用。
    """
    syms = [s.strip().upper() for s in (tickers or list(DEFAULT_TICKERS)) if s.strip()]
    day = cache_date or date.today().isoformat()
    cache = prediction_cache if prediction_cache is not None else {}

    t0 = time.perf_counter()
    rows_by: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_scan_one_ticker, sym, cache, day) for sym in syms]
        for fut in as_completed(futures):
            sym, row = fut.result()
            rows_by[sym] = row

    rows = [rows_by[s] for s in syms if s in rows_by]
    elapsed = time.perf_counter() - t0

    rows.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(rows, start=1):
        r["rank"] = i

    logger.info(
        "スキャン完了: %s銘柄, %.2f秒, cache_size=%s",
        len(rows),
        elapsed,
        len(cache),
    )
    return rows, elapsed


def cache_date_key() -> str:
    return date.today().isoformat()
