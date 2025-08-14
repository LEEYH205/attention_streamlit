@echo off
chcp 65001 >nul

echo 🎯 Attention Streamlit 데모를 시작합니다...

REM 가상환경 활성화
echo 📦 가상환경을 활성화합니다...
call .venv\Scripts\activate.bat

REM 의존성 확인
echo 🔍 필요한 패키지들을 확인합니다...
pip list | findstr /R "streamlit numpy matplotlib torch google-generativeai"

REM Streamlit 앱 실행
echo 🚀 Streamlit 앱을 시작합니다...
echo 📍 브라우저에서 http://localhost:8501 로 접속하세요
echo ⏹️  종료하려면 Ctrl+C를 누르세요
echo.

streamlit run app.py

pause
