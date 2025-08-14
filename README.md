
# Attention 교육용 Streamlit

## 기능
- **단계별 어텐션**: Q,K,V 산출 → 점수/소프트맥스 → 가중합까지 테이블 및 히트맵
- **어텐션 지도**: 타깃(행) × 소스(열) 크로스 어텐션 히트맵
- **임베딩 분석**: 코사인 유사도 히트맵, PCA 2D 분포
- **PyTorch 구현**: SDPA/멀티헤드 최소 예제 코드 + 즉석 형상 테스트
- **AI 챗봇**: 간단 지식베이스에서 쿼리와의 어텐션 유사도 기반 응답(교육용)

## 입력/옵션
- 원본문장(한국어), 번역문장(영어)
- 체크박스: **수식 표시**, **계산 과정**
- 버튼: **분석 시작**

## 설치 & 실행
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 테스트(간단)
- PyTorch 탭에서 **PyTorch 테스트 실행** 버튼 클릭 → 출력 텐서 형상 확인
- 임베딩/어텐션 히트맵이 정상 출력되는지 확인

## 주의
- 임베딩은 해시 기반 임시 벡터와 위치인코딩을 합산한 **교육용** 구현입니다.
- 실제 번역 모델의 내부 가중치와는 다릅니다.
