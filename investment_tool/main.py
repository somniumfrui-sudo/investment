"""米国株 並列比較型投資分析 — Streamlit UI（単一銘柄 / スキャン）。"""

from __future__ import annotations

import inspect
from datetime import date

import pandas as pd
import streamlit as st
import yfinance as yf

from analyzer import analyze_chart
from backtest import run_backtest
from judge import judge
from journal import get_stats, save_top3_buy_from_scan, update_results
from predictor import predict_direction, retrain_from_journal
from scanner import DEFAULT_TICKERS, scan_tickers
from sentiment import analyze_sentiment
from walkforward import run_walkforward

SIGNAL_JP = {"buy": "買い", "neutral": "中立", "sell": "売り"}

WF_CACHE_KEY = "walkforward_results"
SCAN_CACHE_KEY = "scanner_rank_cache"
SCAN_PRED_CACHE_DATE_KEY = "scan_pred_cache_date"
SCAN_PRED_CACHE_BUCKET_KEY = "scan_predict_cache"


def _walkforward_cache_key(ticker: str, train_m: int, test_m: int, total_m: int) -> str:
    return f"{ticker}|{train_m}|{test_m}|{total_m}"


def _badge_color(verdict: str) -> str:
    if verdict == "BUY":
        return "🟢"
    if verdict == "SELL":
        return "🔴"
    return "🟡"


def _latest_price(ticker: str) -> float | None:
    try:
        df = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
        if df is not None and not df.empty:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    return None


def _render_investment_plan(
    ticker: str,
    budget: float,
    verdict: str,
    buy_count: int,
    ai_res: dict,
) -> None:
    if verdict != "BUY" or buy_count < 3:
        return
    if not ai_res.get("both_long_horizons_buy"):
        return

    price = _latest_price(ticker)
    if price is None or price <= 0:
        st.warning("現在価格を取得できなかったため、投資具体案を表示できません。")
        return

    st.subheader("投資具体案（参考）")
    st.caption("※ 3日後・5日後のAIが両方BUYのときのみ表示。実際の取引手数料・スプレッドは含みません。")

    shares_frac = budget / price

    if price > 100:
        st.markdown(
            f"""
**株価が $100 を超えているため、端株（フラクショナルシェア）での購入を案内します。**

- 想定株価: **${price:,.2f}**
- 予算 **${budget:,.2f}** で約 **{shares_frac:.4f} 株** 相当（概算）
"""
        )
    else:
        whole = int(budget // price)
        cost_whole = whole * price
        remainder = budget - cost_whole
        frac_part = remainder / price if remainder > 0 and price > 0 else 0.0
        st.markdown(
            f"""
**株価が $100 以下のため、整数株と端株の両方を表示します。**

- 想定株価: **${price:,.2f}**
- 予算 **${budget:,.2f}** の場合:
  - **整数株**: {whole} 株（約 **${cost_whole:,.2f}**）
  - **端株**: 残り **${remainder:,.2f}** で約 **{frac_part:.4f} 株**（+ 整数株と併せた総株数は約 **{whole + frac_part:.4f} 株**）
"""
        )


def _ai_horizon_table(ai_res: dict) -> pd.DataFrame:
    rows_out = []
    for h in sorted((ai_res.get("horizons") or {}).keys()):
        block = ai_res["horizons"][h]
        rows_out.append(
            {
                "予測期間": block.get("label", str(h)),
                "上昇確率": f"{block['up_probability']*100:.1f}%" if block.get("up_probability") is not None else "—",
                "シグナル": block.get("signal_display", "—"),
            }
        )
    return pd.DataFrame(rows_out)


def render_single_stock_tab() -> None:
    col_in1, col_in2, col_in3 = st.columns([2, 2, 1])
    with col_in1:
        ticker = st.text_input("銘柄コード（例: AAPL）", value="AAPL").strip().upper()
    with col_in2:
        budget = st.slider("予算（USD）", min_value=10, max_value=500, value=100, step=5)
    with col_in3:
        st.write("")
        st.write("")
        run_btn = st.button("分析する", type="primary")

    if not run_btn:
        st.info("銘柄と予算を選んで「分析する」を押してください。")
        return

    if not ticker:
        st.error("銘柄コードを入力してください。")
        return

    wf_train_m, wf_test_m, wf_total_m = 6, 1, 18
    wf_key = _walkforward_cache_key(ticker, wf_train_m, wf_test_m, wf_total_m)
    if WF_CACHE_KEY not in st.session_state:
        st.session_state[WF_CACHE_KEY] = {}
    wf_cache: dict[str, dict] = st.session_state[WF_CACHE_KEY]

    with st.spinner(
        "各モジュールを実行しています…（ウォークフォワードは初回1〜2分かかることがあります）"
    ):
        chart_res = analyze_chart(ticker)
        sent_res = analyze_sentiment(ticker)
        bt_res = run_backtest(ticker)

        if wf_key not in wf_cache:
            wf_cache[wf_key] = run_walkforward(
                ticker,
                train_months=wf_train_m,
                test_months=wf_test_m,
                total_months=wf_total_m,
            )
            st.session_state[WF_CACHE_KEY] = wf_cache

        wf = wf_cache[wf_key]
        cal_by = wf.get("calibration_rows_by_horizon") if wf.get("by_horizon") else None
        ai_res = predict_direction(ticker, calibration_rows_by_horizon=cal_by)

    verdict, buy_count, sell_count, scores = judge(
        chart_res["signal"],
        sent_res["signal"],
        bt_res["signal"],
        ai_res["signal"],
    )

    st.markdown("---")
    b1, b2, b3 = st.columns([1, 2, 2])
    with b1:
        st.metric("総合判定", f"{_badge_color(verdict)} {verdict}")
    with b2:
        st.metric("買いシグナル数", f"{buy_count} / 4")
    with b3:
        st.metric("売りシグナル数", f"{sell_count} / 4")

    st.caption(
        f"モジュール一致: **買い {buy_count}/4** · **売り {sell_count}/4** "
        f"（総合は3票以上でBUY/SELL）・ AIは**3日後・5日後が両方買い**のときだけ「買い」票"
    )

    st.subheader("モジュール別の根拠")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("**① チャート**")
        d = chart_res.get("details") or {}
        st.write(f"シグナル: **{SIGNAL_JP.get(chart_res['signal'], chart_res['signal'])}**")
        st.write(f"強度: **{chart_res.get('strength', 0):+.2f}**（-1〜+1）")
        if chart_res.get("error"):
            st.warning(chart_res["error"])
        else:
            st.caption("指標別")
            for r in d.get("reasons", []):
                st.caption(f"· {r}")

    with c2:
        st.markdown("**② ニュース感情**")
        st.write(f"シグナル: **{SIGNAL_JP.get(sent_res['signal'], sent_res['signal'])}**")
        st.caption(f"手法: {sent_res.get('method', '-')}")
        avg = sent_res.get("avg_scores") or {}
        st.caption(
            f"平均スコア — pos:{avg.get('positive', 0):.3f} neg:{avg.get('negative', 0):.3f} "
            f"neu:{avg.get('neutral', 0):.3f}"
        )
        if sent_res.get("reason"):
            st.caption(sent_res["reason"])
        if sent_res.get("titles"):
            st.caption("直近ニュース見出し（最大5件）:")
            for t in sent_res["titles"]:
                st.caption(f"· {t[:120]}{'…' if len(t) > 120 else ''}")

    with c3:
        st.markdown("**③ バックテスト**")
        st.write(f"シグナル: **{SIGNAL_JP.get(bt_res['signal'], bt_res['signal'])}**")
        if bt_res.get("error"):
            st.warning(bt_res["error"])
        else:
            wr = bt_res.get("win_rate_pct")
            st.caption(
                f"勝率: **{wr:.1f}%**（{bt_res.get('wins', 0)}勝 / {bt_res.get('total_trades', 0)}トレード）"
                if wr is not None
                else "勝率: データなし"
            )
            st.caption(f"最大DD: **{bt_res.get('max_drawdown_pct', 0):.2f}%**")
            st.caption(bt_res.get("reason", ""))

    with c4:
        st.markdown("**④ AI予測（アンサンブル）**")
        if ai_res.get("error"):
            st.warning(ai_res["error"])
        else:
            tbl = _ai_horizon_table(ai_res)
            if not tbl.empty:
                st.dataframe(tbl, use_container_width=True, hide_index=True)
            st.caption(ai_res.get("model_note", ""))
            if ai_res.get("ma20_uptrend"):
                st.caption(
                    "20日MAは**右肩上がり** → SELL 条件でもトレンド尊重で **HOLD** に格下げするルールを適用"
                )
            for h in sorted((ai_res.get("horizons") or {}).keys()):
                blk = ai_res["horizons"][h]
                lbl = blk.get("label", str(h))
                if blk.get("buy_threshold_source") == "walkforward" and blk.get("optimal_buy_threshold_pct") is not None:
                    st.caption(f"{lbl} 最適BUY閾値: **{blk['optimal_buy_threshold_pct']:.1f}%**（ウォークフォワード）")
                elif blk.get("optimal_buy_threshold_pct") is not None:
                    st.caption(f"{lbl} BUY閾値: **{blk['optimal_buy_threshold_pct']:.1f}%**（デフォルト）")
                hd = blk.get("holdout_days")
                if hd:
                    st.caption(f"{lbl} ホールドアウト: 末尾 **{hd}営業日** 固定")
                st.caption(blk.get("reason", ""))

    st.markdown("---")
    st.subheader("バックテスト詳細")
    if bt_res.get("total_trades", 0) == 0 and not bt_res.get("error"):
        st.write("クローズされたトレードがありません。")
    else:
        wr = bt_res.get("win_rate_pct")
        st.write(f"- **勝率**: {wr:.1f} %" if wr is not None else "- **勝率**: —")
        st.write(f"- **総トレード数**: {bt_res.get('total_trades', 0)}")
        st.write(f"- **最大ドローダウン**: {bt_res.get('max_drawdown_pct', 0):.2f} %")

    st.markdown("---")
    _render_investment_plan(ticker, float(budget), verdict, buy_count, ai_res)

    st.markdown("---")
    st.subheader(f"⑤ ウォークフォワード検証（過去{wf_total_m}ヶ月・1/3/5営業日）")
    st.caption(
        "各ホライズンで学習・検証。BUY閾値は50〜75%でキャリブレーション後、MA20上昇時はSELL→HOLD。"
    )

    wf = wf_cache[wf_key]
    if wf.get("error") and not wf.get("by_horizon"):
        st.warning(wf["error"])
    else:
        if wf.get("error"):
            st.info(wf["error"])

        comp = wf.get("comparison_table") or []
        if comp:
            st.markdown("**BUY正解率の比較（翌日 vs 3日後 vs 5日後）**")
            st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)

        by_h = wf.get("by_horizon") or {}
        for h in (1, 3, 5):
            if h not in by_h:
                continue
            blk = by_h[h]
            lbl = blk.get("label", str(h))
            with st.expander(f"{lbl} · 詳細メトリクス・月別・履歴"):
                obt_wf = blk.get("optimal_buy_threshold_pct")
                if obt_wf is not None:
                    st.caption(f"キャリブレーション済み BUY 閾値: **{obt_wf:.1f}%**")
                oa = blk.get("overall_accuracy")
                o_ok, o_n = blk.get("overall_correct", 0), blk.get("overall_total", 0)
                ba, b_ok, b_n = blk.get("buy_accuracy"), blk.get("buy_correct", 0), blk.get("buy_total", 0)
                sa, s_ok, s_n = blk.get("sell_accuracy"), blk.get("sell_correct", 0), blk.get("sell_total", 0)
                c1, c2, c3 = st.columns(3)
                with c1:
                    if oa is not None and o_n:
                        st.metric("全体（BUY+SELL）", f"{oa*100:.1f}%", f"{o_ok}/{o_n}")
                    else:
                        st.metric("全体（BUY+SELL）", "—")
                with c2:
                    if ba is not None and b_n:
                        st.metric("BUY正解率", f"{ba*100:.1f}%", f"{b_ok}/{b_n}")
                    else:
                        st.metric("BUY正解率", "—")
                with c3:
                    if sa is not None and s_n:
                        st.metric("SELL正解率", f"{sa*100:.1f}%", f"{s_ok}/{s_n}")
                    else:
                        st.metric("SELL正解率", "—")

                mdf = blk.get("monthly")
                if isinstance(mdf, pd.DataFrame) and not mdf.empty and mdf.get("accuracy_pct") is not None:
                    if mdf["accuracy_pct"].notna().any():
                        plot_df = mdf.loc[mdf["accuracy_pct"].notna(), ["month", "accuracy_pct"]].copy()
                        plot_df = plot_df.set_index("month").rename(columns={"accuracy_pct": "正解率（%）"})
                        st.line_chart(plot_df)

                hist = blk.get("history_table") or []
                if hist:
                    st.dataframe(pd.DataFrame(hist), use_container_width=True, hide_index=True)


def render_scanner_tab() -> None:
    st.markdown("🔍 **本日のスキャン**（デフォルト10銘柄）")
    budget_scan = st.slider("予算（USD）", min_value=10, max_value=500, value=100, step=5, key="scan_budget")

    tick_str = ", ".join(DEFAULT_TICKERS)
    st.caption(
        f"対象: {tick_str} ・ AIは **1年データ・軽量モデル・CV省略・当日キャッシュ** で高速化（並列4）"
    )

    today = date.today().isoformat()
    if st.session_state.get(SCAN_PRED_CACHE_DATE_KEY) != today:
        st.session_state[SCAN_PRED_CACHE_DATE_KEY] = today
        st.session_state[SCAN_PRED_CACHE_BUCKET_KEY] = {}
    pred_bucket: dict = st.session_state[SCAN_PRED_CACHE_BUCKET_KEY]

    scan_btn = st.button("スキャン実行", type="primary", key="scan_run")

    cache_key = f"{today}|{','.join(DEFAULT_TICKERS)}"
    cached = st.session_state.get(SCAN_CACHE_KEY)

    rows = None
    elapsed_sec: float | None = None
    if scan_btn:
        if cached and cached.get("key") == cache_key:
            rows = cached["rows"]
            elapsed_sec = cached.get("elapsed_sec")
            st.success("当日キャッシュのランキングを表示しています（全銘柄の再分析なし）。")
            print(f"[scanner-ui] ランキングキャッシュヒット key={cache_key}", flush=True)
        else:
            with st.spinner("スキャン中…（目安3分以内・初回のみモデル学習）"):
                _sig = inspect.signature(scan_tickers).parameters
                _kw: dict = {}
                if "prediction_cache" in _sig:
                    _kw["prediction_cache"] = pred_bucket
                if "cache_date" in _sig:
                    _kw["cache_date"] = today
                if "max_workers" in _sig:
                    _kw["max_workers"] = 4
                _out = scan_tickers(list(DEFAULT_TICKERS), **_kw)
                if isinstance(_out, tuple) and len(_out) == 2:
                    rows, elapsed_sec = _out
                else:
                    rows = _out  # 旧 scanner（list のみ返却）
                    elapsed_sec = None
                    st.warning(
                        "古い `scanner.py` が読み込まれています。Streamlit を一度停止して "
                        "`investment_tool` で再起動してください（高速化オプションは無効です）。"
                    )
            st.session_state[SCAN_CACHE_KEY] = {
                "key": cache_key,
                "rows": rows,
                "elapsed_sec": elapsed_sec,
            }
            st.success(
                f"スキャン完了: **{elapsed_sec / 60:.2f}分**（**{elapsed_sec:.1f}秒**）"
                if elapsed_sec is not None
                else "スキャン完了。"
            )
            if elapsed_sec is not None:
                print(
                    f"[scanner-ui] フルスキャン完了 elapsed={elapsed_sec:.2f}s AIキャッシュ条数={len(pred_bucket)}",
                    flush=True,
                )
            else:
                print("[scanner-ui] フルスキャン完了（所要時間未計測）", flush=True)
    elif cached and cached.get("key") == cache_key:
        rows = cached["rows"]
        elapsed_sec = cached.get("elapsed_sec")
        st.caption("※ 本日すでにスキャン済みです。「スキャン実行」で再表示できます。")

    if rows is None:
        if not scan_btn:
            st.info("「スキャン実行」で全銘柄を一括分析します。")
        return

    if elapsed_sec is not None:
        st.metric("直近スキャン所要時間", f"{elapsed_sec:.1f} 秒", help="初回フル実行時。同日再スキャンはキャッシュで短縮されます。")

    display = []
    for r in rows:
        display.append(
            {
                "順位": f"{r['rank']}位",
                "銘柄": r["ticker"],
                "総合スコア": round(r["score"], 2),
                "翌日予測": f"{r.get('prob_1d', 0)*100:.0f}% {r.get('pred_1d', '—')}",
                "3日予測": f"{r['prob_3d']*100:.0f}% {r['pred_3d']}",
                "5日予測": f"{r['prob_5d']*100:.0f}% {r['pred_5d']}",
                "チャート": r["chart_jp"],
                "感情": r["sent_jp"],
            }
        )
    st.dataframe(pd.DataFrame(display), use_container_width=True, hide_index=True)

    if st.button("シグナルを日誌に保存", type="secondary", key="journal_save_scan"):
        n = save_top3_buy_from_scan(rows, float(budget_scan))
        if n:
            st.success(f"日誌に **{n}** 件保存しました（journal.csv）。BUY の期間のみ記録されます。")
        else:
            st.warning("保存対象がありません（上位3銘柄に BUY シグナルが無いか、価格取得に失敗しました）。")

    st.markdown("**上位3銘柄の投資具体案（参考）**")
    top3 = [r["ticker"] for r in rows[:3]]
    for sym in top3:
        st.markdown(f"##### {sym}")
        price = _latest_price(sym)
        if price is None or price <= 0:
            st.warning(f"{sym}: 価格を取得できませんでした。")
            continue
        bud = float(budget_scan)
        sh = bud / price
        if price > 100:
            st.write(f"- 端株のみ案内: 約 **{sh:.4f} 株**（${price:,.2f}・予算 ${bud:,.2f}）")
        else:
            w = int(bud // price)
            st.write(f"- 整数株 {w} 株 + 端株の組み合わせ可能（株価 ${price:,.2f}）")


def render_journal_tab() -> None:
    st.markdown("📓 **トレード日誌**（`journal.csv` に蓄積・自動照合）")
    with st.spinner("未確定レコードを照合中…"):
        n_up = update_results()
    if n_up:
        st.caption(f"今回の更新: **{n_up}** 件の結果を確定しました。")

    stats = get_stats()
    if stats.get("empty"):
        st.info("まだ記録がありません。スキャンタブで「シグナルを日誌に保存」を押してください。")
        return

    st.subheader("📊 トレード実績サマリー")
    ovr = stats.get("overall_hit_rate")
    bt = stats.get("best_ticker")
    bp = stats.get("best_period")
    tp = stats.get("total_pnl")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "全体的中率",
            f"{ovr*100:.1f}%" if ovr is not None else "—",
            help=f"確定 {stats.get('resolved_n', 0)} 件",
        )
    with c2:
        if bt:
            st.metric("銘柄別1位", f"{bt[0]}", f"{bt[1]['rate']*100:.1f}% 的中")
        else:
            st.metric("銘柄別1位", "—")
    with c3:
        if bp:
            st.metric("期間別1位", f"{bp[0]}", f"{bp[1]['rate']*100:.1f}% 的中")
        else:
            st.metric("期間別1位", "—")
    with c4:
        st.metric("総損益（確定）", f"${tp:+.2f}" if tp is not None else "—")
    st.caption(f"未確定（待機中）: **{stats.get('pending_n', 0)}** 件")

    st.subheader("AI確率帯別の的中率")
    st.caption("AI確率が高いほど当たっているか？")
    for b in stats.get("bucket_stats") or []:
        r = b.get("rate")
        n = b.get("n", 0)
        if r is None or n == 0:
            st.write(f"**{b['bucket']}** — データなし")
        else:
            st.write(f"**{b['bucket']}**　{r*100:.0f}% 的中（{n}件）")

    st.subheader("月別損益")
    mp = stats.get("monthly_pnl")
    if isinstance(mp, pd.DataFrame) and not mp.empty:
        plot_m = mp.set_index("month").rename(columns={"pnl": "損益（USD）"})
        st.bar_chart(plot_m)
    else:
        st.caption("損益データがありません。")

    st.subheader("全トレード履歴")
    raw = stats.get("raw")
    if isinstance(raw, pd.DataFrame) and not raw.empty:
        disp = raw.copy()
        def _fmt_hit(v) -> str:
            try:
                h = int(float(v))
            except (TypeError, ValueError):
                return "—"
            return {1: "✅ 的中", 0: "❌ 外れ", -1: "⏳ 待機中"}.get(h, "—")

        disp["的中表示"] = disp["hit"].map(_fmt_hit)
        def _fmt_ret(r: pd.Series) -> str:
            try:
                h = int(float(r.get("hit", -1)))
            except (TypeError, ValueError):
                h = -1
            if h == -1:
                return "未確定"
            if pd.notna(r.get("actual_return")):
                return f"{float(r['actual_return']):+.2f}%"
            return "—"

        def _fmt_pnl(r: pd.Series) -> str:
            try:
                h = int(float(r.get("hit", -1)))
            except (TypeError, ValueError):
                h = -1
            if h == -1 or pd.isna(r.get("pnl")):
                return "—"
            return f"${float(r['pnl']):+.2f}"

        disp["実際結果"] = disp.apply(_fmt_ret, axis=1)
        disp["損益表示"] = disp.apply(_fmt_pnl, axis=1)
        show = disp[
            [
                "date",
                "ticker",
                "period",
                "signal",
                "ai_prob",
                "実際結果",
                "損益表示",
                "的中表示",
            ]
        ].rename(
            columns={
                "date": "日付",
                "ticker": "銘柄",
                "period": "期間",
                "signal": "シグナル",
                "ai_prob": "AI確率(%)",
            }
        )
        st.dataframe(show, use_container_width=True, hide_index=True)
        csv_bytes = raw.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "CSVをダウンロード",
            data=csv_bytes,
            file_name="journal_export.csv",
            mime="text/csv",
            key="dl_journal",
        )
    else:
        st.caption("表示する行がありません。")

    st.subheader("モデル再学習（日誌データ）")
    st.caption("確定済みのトレードから特徴量を復元し、各ホライズンの学習データに追加します（journal_boost.npz）。")
    if st.button("蓄積データで精度を改善する", type="primary", key="journal_retrain"):
        n_add, msg = retrain_from_journal()
        if n_add > 0:
            st.success(msg)
        else:
            st.warning(msg)


def main() -> None:
    st.set_page_config(page_title="米国株 並列分析", layout="wide")
    st.title("米国株 投資分析ツール（並列比較型）")
    st.caption("4モジュールが独立してシグナルを出し、多数決で総合判定します。")

    tab_single, tab_scan, tab_journal = st.tabs(
        ["単一銘柄分析", "銘柄スキャン（ランキング）", "トレード日誌"]
    )

    with tab_single:
        render_single_stock_tab()

    with tab_scan:
        render_scanner_tab()

    with tab_journal:
        render_journal_tab()

    st.markdown("---")
    st.warning(
        "このツールは情報提供目的です。投資判断はご自身の責任で行ってください。"
    )


if __name__ == "__main__":
    main()
