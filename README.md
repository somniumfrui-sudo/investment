# 米国株 投資分析ツール（Streamlit）

`investment_tool/` の Streamlit アプリを GitHub 経由でクラウド公開する手順です。

## 前提

- **GitHub ↔ このアプリ**: [Streamlit Community Cloud](https://share.streamlit.io/) が最適（無料・GitHub 連携・Python/Streamlit 対応）
- **Vercel**: サーバーレス向けのため、常時起動の Streamlit + `torch` には**非推奨**（タイムアウト・容量制限で動きません）

株価取得（`yfinance`）はインターネット経由です。ローカルで次が通ればデータ取得は可能です。

```powershell
cd c:\investment\investment_tool
python -c "import yfinance as yf; print(yf.Ticker('AAPL').history(period='5d').tail(1))"
```

## 1. GitHub に上げる

PowerShell:

```powershell
cd c:\investment
git init
git add .
git commit -m "Initial commit: investment Streamlit app"
```

GitHub で空リポジトリを作成（例: `investment-tool`）し、表示される URL で:

```powershell
git branch -M main
git remote add origin https://github.com/<あなたのユーザー名>/<リポジトリ名>.git
git push -u origin main
```

初回は GitHub ログイン（ブラウザ or Personal Access Token）が必要です。

## 2. Streamlit Community Cloud（推奨）

1. https://share.streamlit.io/ に GitHub でログイン
2. **New app** → リポジトリを選択
3. 設定例:
   - **Main file path**: `streamlit_app.py`
   - **App URL**: 任意
4. **Deploy**（初回は `requirements.txt` のインストールに 5〜15 分かかることがあります）

`torch` / `transformers` はメモリを多く使うため、無料枠でビルド失敗する場合は `requirements.txt` から該当行を外し、感情分析をオフにするなどの軽量化を検討してください。

## 3. Vercel について

このリポジトリは **Streamlit 専用**です。Vercel で同じ UI を動かすには Next.js 等への作り直しが必要です。

クラウドで URL を1つにしたい場合の代替:

| サービス | 用途 |
|---------|------|
| Streamlit Cloud | このアプリ向け（推奨） |
| Render / Railway | Docker で Streamlit を常時起動 |
| Vercel | 静的サイト・Next.js API のみ |

## ローカル起動

```powershell
cd c:\investment\investment_tool
pip install -r ..\requirements.txt
streamlit run main.py
```

またはルートから:

```powershell
cd c:\investment
streamlit run streamlit_app.py
```
