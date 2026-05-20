# 米国株 投資分析ツール（Streamlit）

- **GitHub** … コードの保存・同期（ブラウザでは**動きません**）
- **Streamlit Cloud** … インターネットからアプリを開く（`https://xxxx.streamlit.app`）

## ネットで見る（初回だけ・約5分）

1. https://share.streamlit.io/ を開く
2. **Continue with GitHub** でログイン
3. **New app** → リポジトリ `somniumfrui-sudo/investment` を選ぶ
4. 設定:
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
5. **Deploy** を押す（初回は依存関係のインストールで 5〜15 分）

完了すると **https://（あなたが付けた名前）.streamlit.app** の URL が表示されます。  
GitHub のページ（`github.com/.../investment`）を開いてもアプリは起動しません。

> クラウド版はメモリ節約のため FinBERT（`torch`）は使わず、感情分析は TextBlob になります。ローカルでは従来どおり FinBERT も利用可能です。

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
