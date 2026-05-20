"""並列モジュールの多数決で総合判定を行う。"""


def judge(chart, sentiment, backtest, ai):
    scores = {
        "chart": chart,
        "sentiment": sentiment,
        "backtest": backtest,
        "ai": ai,
    }

    buy_count = sum(1 for v in scores.values() if v == "buy")
    sell_count = sum(1 for v in scores.values() if v == "sell")

    if buy_count >= 3:
        verdict = "BUY"
    elif sell_count >= 3:
        verdict = "SELL"
    else:
        verdict = "HOLD"

    return verdict, buy_count, sell_count, scores
