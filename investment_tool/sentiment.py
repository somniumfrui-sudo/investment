"""ニュース見出しの感情分析（FinBERT優先、失敗時はTextBlob）。"""

from __future__ import annotations

import numpy as np
import yfinance as yf
from textblob import TextBlob


def _textblob_scores(text: str) -> dict[str, float]:
    polarity = float(TextBlob(text).sentiment.polarity)  # -1 .. 1
    pos = max(0.0, polarity)
    neg = max(0.0, -polarity)
    neu = 1.0 - abs(polarity)
    neu = max(0.0, neu)
    s = pos + neg + neu
    if s > 0:
        pos, neg, neu = pos / s, neg / s, neu / s
    else:
        pos = neg = 0.0
        neu = 1.0
    return {"positive": pos, "negative": neg, "neutral": neu}


def _finbert_scores_batch(titles: list[str]) -> list[dict[str, float]] | None:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
    except Exception:
        return None

    model_name = "ProsusAI/finbert"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
    except Exception:
        return None

    id2label = getattr(model.config, "id2label", None) or {
        0: "positive",
        1: "negative",
        2: "neutral",
    }

    results: list[dict[str, float]] = []
    with torch.no_grad():
        for title in titles:
            enc = tokenizer(
                title,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            logits = model(**enc).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            label_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            for i, p in enumerate(probs):
                lab = id2label.get(i, id2label.get(str(i), i))
                lab = str(lab).lower()
                if lab == "positive":
                    label_scores["positive"] += float(p)
                elif lab == "negative":
                    label_scores["negative"] += float(p)
                else:
                    label_scores["neutral"] += float(p)
            results.append(label_scores)
    return results


def analyze_sentiment(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    news = getattr(t, "news", None)
    if news is None:
        news = []

    titles: list[str] = []
    for item in news[:5]:
        if isinstance(item, dict) and item.get("title"):
            titles.append(str(item["title"]))

    if not titles:
        return {
            "signal": "neutral",
            "method": "none",
            "avg_scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
            "titles": [],
            "reason": "ニュースが取得できませんでした。",
        }

    method = "finbert"
    per_title_scores: list[dict[str, float]] | None = _finbert_scores_batch(titles)

    if per_title_scores is None:
        method = "textblob"
        per_title_scores = [_textblob_scores(tt) for tt in titles]

    avg_pos = float(np.mean([s["positive"] for s in per_title_scores]))
    avg_neg = float(np.mean([s["negative"] for s in per_title_scores]))
    avg_neu = float(np.mean([s["neutral"] for s in per_title_scores]))

    # 平均スコアが最大のラベルを採用し、buy/sell にマッピング
    if avg_pos >= avg_neg and avg_pos >= avg_neu:
        signal = "buy"
        reason = f"positive が最大（平均 {avg_pos:.3f}）"
    elif avg_neg >= avg_pos and avg_neg >= avg_neu:
        signal = "sell"
        reason = f"negative が最大（平均 {avg_neg:.3f}）"
    else:
        signal = "neutral"
        reason = f"neutral が最大（平均 {avg_neu:.3f}）"

    return {
        "signal": signal,
        "method": method,
        "avg_scores": {"positive": avg_pos, "negative": avg_neg, "neutral": avg_neu},
        "per_title": list(zip(titles, per_title_scores)),
        "titles": titles,
        "reason": reason,
    }
