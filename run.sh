#!/bin/bash

# Attention Streamlit 데모 실행 스크립트
echo "🎯 Attention Streamlit 데모를 시작합니다..."

# 가상환경 활성화
echo "📦 가상환경을 활성화합니다..."
source .venv/bin/activate

# 의존성 확인
echo "🔍 필요한 패키지들을 확인합니다..."
pip list | grep -E "(streamlit|numpy|matplotlib|torch|google-generativeai)"

# Streamlit 앱 실행
echo "🚀 Streamlit 앱을 시작합니다..."
echo "📍 브라우저에서 http://localhost:8501 로 접속하세요"
echo "⏹️  종료하려면 Ctrl+C를 누르세요"
echo ""

streamlit run app.py
