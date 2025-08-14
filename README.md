
# 🎯 Attention Mechanism 교육용 Streamlit 데모 v2.0

어텐션 메커니즘의 핵심을 단계별로 시각화하여 보여주는 교육용 웹 애플리케이션입니다. 초보자부터 고급 사용자까지 어텐션의 원리를 체계적으로 학습할 수 있습니다.

## ✨ 주요 기능

### 🧭 학습 가이드 (초보자용)
- **7단계 체계적 학습**: 토큰화부터 최종 컨텍스트 벡터까지
- **직관적 설명**: Q, K, V의 역할을 도서관 비유로 이해
- **수치 안정성 데모**: softmax 오버플로우 방지 원리
- **단계별 수식**: 각 단계의 수학적 표현과 의미

### 🔬 어텐션 실험실
- **3가지 어텐션 유형**: Self-Attention, Masked Self-Attention, Cross-Attention
- **실시간 분석**: 선택한 유형에 따른 즉시 결과 확인
- **상세한 과정**: 각 단계별 계산 과정과 해석

### 🧩 멀티헤드 시각화
- **헤드 수 조절**: 1~8개 헤드로 실험
- **개별 헤드 분석**: 각 헤드의 어텐션 패턴 시각화
- **헤드 간 비교**: 평균 어텐션과 엔트로피 분석
- **패턴 요약**: 각 헤드의 주요 특징 정리

### ⛔ 마스킹 & 인과성
- **Causal Mask Matrix**: 시각적 마스킹 매트릭스
- **수학적 표현**: `M_{ij} = 1[j ≤ i]` 수식 설명
- **마스킹 전후 비교**: 실제 적용 효과 확인
- **인과성의 중요성**: 학습과 추론에서의 역할

### 📊 어텐션 지도 & 분석
- **히트맵 시각화**: 어텐션 가중치의 직관적 표현
- **통계 분석**: 평균, 분산, 엔트로피 계산
- **패턴 분석**: 집중 vs 분산 어텐션 구분

### 📐 임베딩 분석
- **코사인 유사도**: 토큰 간 관계 분석
- **PCA 2D 분포**: 고차원 임베딩의 시각화

### 🛠️ PyTorch 구현
- **실제 코드**: Scaled Dot-Product Attention 구현
- **텐서 연산**: numpy vs PyTorch 비교

### 💬 AI 챗봇 (Gemini AI)
- **Gemini AI 연결**: Google의 최신 AI 모델과 대화
- **분석 결과 기반**: 실제 계산된 데이터로 컨텍스트 제공
- **교육적 답변**: 어텐션 메커니즘에 대한 상세한 설명
- **대화 히스토리**: 질문-답변 기록 저장 및 관리
- **폴백 지원**: API 키가 없을 때 키워드 기반 응답 제공

### 📝 퀴즈
- **3문항 점검**: 핵심 개념 즉시 확인
- **정답 피드백**: 오답에 대한 상세한 설명
- **점수 계산**: 학습 진도 파악

### 📚 용어사전
- **검색 기능**: 원하는 용어 빠른 찾기
- **한글/영문 병기**: 접근성 향상
- **상세 설명**: 개념의 핵심과 예시

## 🚀 시작하기

### 1. 환경 설정
```bash
# 저장소 클론
git clone <repository-url>
cd Attention_streamlit

# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# 또는
.venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. Gemini API 설정 (선택사항)
AI 챗봇 기능을 사용하려면 Google Gemini API 키가 필요합니다:

```bash
# 방법 1: config.env 파일 사용 (권장)
# config.env 파일을 편집하여 실제 API 키 입력
GOOGLE_API_KEY=your_actual_api_key_here

# 방법 2: 환경 변수 설정
export GOOGLE_API_KEY="your_api_key_here"  # macOS/Linux
# 또는
set GOOGLE_API_KEY=your_api_key_here       # Windows

# 방법 3: .env 파일 생성
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

**API 키 발급 방법:**
1. [Google AI Studio](https://aistudio.google.com/app/apikey) 접속
2. "Create API Key" 클릭
3. 생성된 키를 복사하여 위의 방법 중 하나로 설정

**config.env 파일 사용법 (권장):**
1. `config.env` 파일을 텍스트 에디터로 열기
2. `GOOGLE_API_KEY=your_gemini_api_key_here` 부분을 실제 API 키로 교체
3. 파일 저장
4. 앱 실행 시 자동으로 로드됨

**⚠️ 보안 주의사항:**
- `config.env` 파일은 `.gitignore`에 포함되어 Git에 커밋되지 않음
- API 키를 공개 저장소에 업로드하지 마세요
- 팀원과 공유할 때는 `config.env.example` 파일을 사용하세요

### 3. 앱 실행

#### 🚀 간편 실행 (권장)
```bash
# macOS/Linux
./run.sh

# Windows
run.bat
```

#### 🔧 수동 실행
```bash
# 가상환경 활성화
source .venv/bin/activate  # macOS/Linux
# 또는
.venv\Scripts\activate     # Windows

# 방법 1: 가상환경의 Python으로 직접 실행 (권장)
.venv/bin/python -m streamlit run app.py  # macOS/Linux
# 또는
.venv\Scripts\python.exe -m streamlit run app.py  # Windows

# 방법 2: 가상환경 활성화 후 실행
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속하세요.

## 📖 사용 가이드

### 🎯 초보자 학습 경로
1. **학습 가이드** → 단계별로 개념 이해
2. **용어사전** → 모르는 용어 검색
3. **실험실** → 다양한 어텐션 유형 실험
4. **퀴즈** → 학습 내용 점검

### 🔬 고급 사용자 활용법
1. **멀티헤드 시각화** → 헤드별 패턴 분석
2. **마스킹 & 인과성** → 수학적 원리 이해
3. **어텐션 지도** → 상세한 통계 분석
4. **PyTorch 구현** → 실제 코드 학습

## 🏗️ 아키텍처

### 핵심 구성 요소
- **토크나이저**: 간단한 공백/구두점 기반 분리
- **임베딩**: 해시 기반 결정적 벡터 생성
- **위치 인코딩**: 사인-코사인 위치 정보
- **어텐션 계산**: Scaled Dot-Product Attention
- **멀티헤드**: 병렬 어텐션 처리

### 기술 스택
- **Frontend**: Streamlit
- **Backend**: Python, NumPy
- **시각화**: Matplotlib
- **딥러닝**: PyTorch (구현 예시)

## 📚 학습 내용

### 어텐션 메커니즘의 7단계
1. **토큰화 & 임베딩**: 텍스트를 벡터로 변환
2. **선형사영(Q,K,V)**: Query, Key, Value 생성
3. **유사도(QKᵀ)**: 어텐션 스코어 계산
4. **스케일링(/√dₖ)**: 수치 안정성 확보
5. **소프트맥스(가중치)**: 확률 분포로 변환
6. **가중합(컨텍스트)**: 최종 출력 계산

### 핵심 개념
- **Self-Attention**: 같은 시퀀스 내 관계 학습
- **Cross-Attention**: 서로 다른 시퀀스 간 관계
- **Masked Attention**: 미래 정보 차단
- **Multi-Head**: 다양한 관점에서 정보 종합

## 🔧 커스터마이징

### 입력 텍스트 변경
사이드바에서 한국어 원본문장과 영어 번역문장을 자유롭게 수정할 수 있습니다.

### 옵션 설정
- **수식 표시**: 수학적 표현 표시/숨김
- **계산 과정**: 상세한 계산 단계 표시/숨김

## 📈 성능 최적화

### 메모리 효율성
- 토큰 수에 따른 동적 차원 조정
- 불필요한 중간 결과 제거

### 계산 최적화
- 벡터화된 연산 사용
- 효율적인 어텐션 계산

## 🤝 기여하기

### 버그 리포트
- GitHub Issues에 상세한 문제 설명
- 재현 가능한 예시 포함

### 기능 제안
- 새로운 시각화 방법
- 추가 학습 콘텐츠
- 성능 개선 방안

### 코드 기여
1. Fork 저장소
2. 기능 브랜치 생성
3. 변경사항 커밋
4. Pull Request 생성

## 📄 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다. 자유롭게 학습과 연구에 활용하세요.

## 🙏 감사의 말

- **Transformer 논문**: Attention Is All You Need
- **Streamlit**: 웹 애플리케이션 프레임워크
- **커뮤니티**: 피드백과 제안을 주신 모든 분들

## 📞 연락처

프로젝트에 대한 질문이나 제안사항이 있으시면 GitHub Issues를 통해 연락해주세요.

---

**🎯 어텐션 메커니즘의 핵심을 체계적으로 학습하고, 실제 구현을 통해 이해를 깊게 하세요!**
