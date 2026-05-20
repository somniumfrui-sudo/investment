# 米国株 投資分析ツール（Streamlit）

**ローカル（localhost）で使う**アプリです。GitHub はコードのバックアップ・同期用（任意）です。

## 起動

**いちばん簡単**: `investment_tool` フォルダの **`start.bat`** をダブルクリック

または PowerShell:

```powershell
cd c:\investment\investment_tool
python -m streamlit run main.py
```

> `streamlit run` だけだと「コマンドが見つからない」と出ることがあります。必ず **`python -m streamlit`** を使ってください。

ブラウザで **http://localhost:8501** を開きます（自動で開かない場合は手入力）。

株価取得の確認:

```powershell
python -c "import yfinance as yf; print(yf.Ticker('AAPL').history(period='5d').tail(1))"
```

## GitHub（任意）

変更を保存したいとき:

```powershell
cd c:\investment
git add .
git commit -m "変更内容"
git push
```

リポジトリ: https://github.com/somniumfrui-sudo/investment
