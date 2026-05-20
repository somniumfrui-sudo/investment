# 米国株 投資分析ツール（Streamlit）

GitHub でコードを管理し、**ローカルで起動**する構成です。

## ローカル起動

```powershell
cd c:\investment\investment_tool
pip install -r requirements.txt
streamlit run main.py
```

株価取得（`yfinance`）はインターネット経由です。接続確認:

```powershell
python -c "import yfinance as yf; print(yf.Ticker('AAPL').history(period='5d').tail(1))"
```

## GitHub への push

```powershell
cd c:\investment
git add .
git commit -m "変更内容のメモ"
git push
```

初回だけ（リポジトリ未作成の場合）:

1. https://github.com/new で空リポジトリ `investment` を作成（README は追加しない）
2. 以下を実行:

```powershell
cd c:\investment
git remote set-url origin https://github.com/<ユーザー名>/investment.git
git push -u origin main
```

別 PC で使うとき:

```powershell
git clone https://github.com/<ユーザー名>/investment.git
cd investment\investment_tool
pip install -r requirements.txt
streamlit run main.py
```
