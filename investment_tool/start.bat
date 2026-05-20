@echo off
cd /d "%~dp0"
echo Starting Streamlit at http://localhost:8501 ...
python -m streamlit run main.py
if errorlevel 1 pause
