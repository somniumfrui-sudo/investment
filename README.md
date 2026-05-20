# 米国株 投資分析ツール（Streamlit）

**ローカル（localhost）で使う**アプリです。GitHub はコードのバックアップ・同期用（任意）です。

## 起動

```powershell
cd c:\investment\investment_tool
pip install -r requirements.txt
streamlit run main.py
```

ブラウザで **http://localhost:8501** が開きます。

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
