
# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
from dataclasses import dataclass
import math
import google.generativeai as genai
import os
from dotenv import load_dotenv

# 탭 모듈들 import
from tabs.learning_guide import render_learning_guide
from tabs.attention_lab import render_attention_lab
from tabs.softmax_lab import render_softmax_lab
from tabs.multihead_visualization import render_multihead_visualization
from tabs.masking_causality import render_masking_causality
from tabs.attention_map import render_attention_map
from tabs.embedding_analysis import render_embedding_analysis

# 한글 폰트 설정
import matplotlib.font_manager as fm
import platform

# 운영체제별 한글 폰트 설정
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:  # Linux
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="Attention 교육용 데모", layout="wide")

# Gemini API 설정
def setup_gemini_api():
    """Gemini API 설정 및 초기화"""
    # config.env 파일 로드
    load_dotenv('config.env')
    
    # API 키 로드 우선순위: 1) config.env, 2) 환경변수, 3) Streamlit secrets
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        api_key = st.secrets.get("GOOGLE_API_KEY") if hasattr(st, 'secrets') else None
    
    if not api_key:
        # 사이드바에 API 키 입력 필드 제공
        api_key = st.sidebar.text_input(
            "🔑 Google Gemini API 키",
            type="password",
            help="Google AI Studio에서 API 키를 발급받아 입력하세요. https://aistudio.google.com/app/apikey"
        )
        
        if api_key:
            # API 키를 세션 상태에 저장
            st.session_state.gemini_api_key = api_key
            st.success("✅ API 키가 설정되었습니다!")
        else:
            st.sidebar.warning("⚠️ Gemini API 기능을 사용하려면 API 키를 입력하세요.")
            return None
    
    try:
        # Gemini API 초기화
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except Exception as e:
        st.error(f"❌ Gemini API 초기화 실패: {str(e)}")
        return None

# Gemini API 모델 초기화
gemini_model = setup_gemini_api()

# -----------------------------
# Utilities
# -----------------------------
def tokenize(text: str):
    """간단 토크나이저: 공백/구두점 기준 분리"""
    import re
    # 단어, 숫자, 아포스트로피/하이픈을 포함하는 토큰 + 그 외 구두점
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9가-힣]+(?:['-][A-Za-z0-9가-힣]+)?|[^\sA-Za-z0-9가-힣]", text.strip())
    return tokens if tokens else ["<empty>"]

def sinusoidal_positional_encoding(n: int, d: int):
    """논문식 사인-코사인 위치인코딩"""
    pe = np.zeros((n, d), dtype=np.float64)
    position = np.arange(n)[:, None]
    div_term = np.exp(np.arange(0, d, 2) * -(np.log(10000.0) / d))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

def hash_vec(token: str, dim: int = 32) -> np.ndarray:
    """토큰별 결정적 해시 임베딩(시연용)"""
    rs = np.random.RandomState(abs(hash(token)) % (2**32))
    return rs.normal(0, 1, size=dim)

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

def calculate_entropy(attention_weights):
    """어텐션 가중치의 엔트로피 계산"""
    # 0에 가까운 값들을 작은 양수로 대체
    eps = 1e-10
    weights = np.maximum(attention_weights, eps)
    # 정규화
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    # 엔트로피 계산: -sum(p * log(p))
    entropy = -np.sum(weights * np.log(weights), axis=-1)
    return np.mean(entropy)  # 평균 엔트로피 반환

def scaled_dot_product_attention(Q, K, V, mask=None):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)  # [tgt, src]
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    context = weights @ V
    return context, weights, scores

def linear_projection(X, W, b=None):
    Y = X @ W
    if b is not None:
        Y += b
    return Y

def np_to_df(mat, row_idx=None, col_idx=None, floatfmt=6):
    import pandas as pd
    df = pd.DataFrame(np.round(mat.astype(float), floatfmt))
    
    # 중복된 인덱스/컬럼 이름 문제 해결
    if row_idx is not None:
        # 중복된 값이 있으면 _숫자 추가
        unique_row_idx = []
        seen = {}
        for i, val in enumerate(row_idx):
            if val in seen:
                seen[val] += 1
                unique_row_idx.append(f"{val}_{seen[val]}")
            else:
                seen[val] = 0
                unique_row_idx.append(val)
        df.index = unique_row_idx
    
    if col_idx is not None:
        # 중복된 값이 있으면 _숫자 추가
        unique_col_idx = []
        seen = {}
        for i, val in enumerate(col_idx):
            if val in seen:
                seen[val] += 1
                unique_col_idx.append(f"{val}_{seen[val]}")
            else:
                seen[val] = 0
                unique_col_idx.append(val)
        df.columns = unique_col_idx
    
    return df

def plot_heatmap(W, xticks, yticks, title=""):
    fig, ax = plt.subplots()
    im = ax.imshow(W)  # 색상은 기본값 사용(규정: 특정 색상 지정 금지)
    ax.set_xticks(np.arange(len(xticks)))
    ax.set_yticks(np.arange(len(yticks)))
    ax.set_xticklabels(xticks, rotation=45, ha="right")
    ax.set_yticklabels(yticks)
    ax.set_title(title, fontsize=12)
    # 값 주석(칸이 너무 많으면 생략)
    if W.shape[0]*W.shape[1] <= 400:
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                ax.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center", fontsize=8)
    st.pyplot(fig)

def pca_2d(X, k=2):
    """numpy SVD 기반 간단 PCA"""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt.T[:, :k]
    return Z

# -----------------------------
# Sidebar (입력/옵션/버튼)
# -----------------------------
st.sidebar.header("입력 & 옵션")
src_text = st.sidebar.text_area("원본문장 (한국어)", "나는 밥을 먹었어")
tgt_text = st.sidebar.text_area("번역문장 (영어)", "I ate a meal")
show_formula = st.sidebar.checkbox("수식 표시", value=True)
show_steps = st.sidebar.checkbox("계산 과정", value=True)
analyze = st.sidebar.button("분석 시작")

st.title("🎯 Attention 교육용 Streamlit 데모")
st.caption("단계별 어텐션 · 멀티헤드 어텐션 · 어텐션 지도 · 임베딩 분석 · PyTorch 구현 · AI 챗봇")

tabs = st.tabs(["🧭 학습 가이드(초보자용)", "🔬 단계별 어텐션(실험실)", "🧪 소프트맥스 실험실", "🧩 멀티헤드 시각화", "⛔ 마스킹 & 인과성", "어텐션 지도", "임베딩 분석", "PyTorch 구현", "AI 챗봇", "📝 퀴즈", "📚 용어사전"])

# 공통 전처리
src_tokens = tokenize(src_text)
tgt_tokens = tokenize(tgt_text)

dim = 32  # 임베딩 차원(시연용)
src_E = np.stack([hash_vec(t, dim) for t in src_tokens])
tgt_E = np.stack([hash_vec(t, dim) for t in tgt_tokens])
src_E = src_E + sinusoidal_positional_encoding(len(src_tokens), dim)
tgt_E = tgt_E + sinusoidal_positional_encoding(len(tgt_tokens), dim)

# 선형 변환 가중치(시연용 고정 난수)
np.random.seed(42)
Wq = np.random.normal(0, 1 / np.sqrt(dim), size=(dim, dim))
Wk = np.random.normal(0, 1 / np.sqrt(dim), size=(dim, dim))
Wv = np.random.normal(0, 1 / np.sqrt(dim), size=(dim, dim))

# --------------------------------------------------
# 탭 1: 학습 가이드(초보자용)
# --------------------------------------------------
with tabs[0]:
    render_learning_guide()
    
# 탭 2: 단계별 어텐션(실험실)
# --------------------------------------------------
with tabs[1]:
    st.subheader("🔬 어텐션 실험실")
    st.markdown("""
    다양한 어텐션 유형을 실험해보세요. 각 유형을 선택하고 분석 결과를 확인할 수 있습니다.
    """)
    
    # 어텐션 유형 선택
    attention_type = st.radio(
        "어텐션 유형 선택",
        ["인코더 Self-Attention", "디코더 Masked Self-Attention", "크로스 Attention"],
        help="각 유형의 특징을 확인해보세요"
    )
    
    # 공식 표시 (항상 표시)
    st.markdown("### 📊 공식")
    st.latex(r"Q = XW_Q,\ K = XW_K,\ V = XW_V")
    st.latex(r"\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V")
    st.caption("※ 여기서는 교육용으로 소스/타깃 임베딩에 같은 차원/가중치를 사용합니다.")
    
    # 토큰 정보 표시
    st.markdown("### 🏷️ 토큰 정보")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**소스 토큰 (한국어):**", src_tokens)
    with c2:
        st.write("**타깃 토큰 (영어):**", tgt_tokens)

    if analyze:
        if attention_type == "인코더 Self-Attention":
            st.markdown("### 🔍 인코더 Self-Attention (소스→소스)")
            st.markdown("**목적**: 한국어 문장 내에서 각 단어가 다른 단어들과 어떻게 연결되는지 학습")
            st.markdown("**특징**: 모든 단어가 서로를 참고할 수 있음 (양방향)")
            
            # A-1. 선형 변환 (Query, Key, Value 생성)
            st.markdown("#### A-1. 선형 변환: Query, Key, Value 생성")
            st.markdown("각 단어 임베딩을 Query, Key, Value로 변환합니다.")
            
            Qs = linear_projection(src_E, Wq)
            Ks = linear_projection(src_E, Wk)
            Vs = linear_projection(src_E, Wv)
            
            # Query, Key, Value 표시
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Query (Q)**")
                st.write("shape:", Qs.shape)
                st.dataframe(np_to_df(Qs, row_idx=src_tokens))
            with col2:
                st.markdown("**Key (K)**")
                st.write("shape:", Ks.shape)
                st.dataframe(np_to_df(Ks, row_idx=src_tokens))
            with col3:
                st.markdown("**Value (V)**")
                st.write("shape:", Vs.shape)
                st.dataframe(np_to_df(Vs, row_idx=src_tokens))
            
            # A-2. 어텐션 스코어 계산
            st.markdown("#### A-2. 어텐션 스코어 계산: QKᵀ/√d")
            st.markdown("각 단어 쌍 간의 유사도를 계산합니다.")
            
            dk = Qs.shape[-1]
            Ss = Qs @ Ks.T / np.sqrt(dk)
            
            # Score Matrix 표시
            st.write("**Score Matrix (QKᵀ/√d)**")
            st.write("shape:", Ss.shape)
            st.dataframe(np_to_df(Ss, row_idx=src_tokens, col_idx=src_tokens))
            st.markdown("**해석**: 값이 클수록 해당 단어 쌍이 더 관련성이 높습니다.")
            
            # A-3. Softmax 적용
            st.markdown("#### A-3. Softmax 적용: 확률 분포로 변환")
            st.markdown("스코어를 확률 분포로 변환하여 어텐션 가중치를 만듭니다.")
            
            Ws = softmax(Ss, axis=-1)
            
            # Softmax Weights 표시
            st.write("**Softmax Weights**")
            st.write("shape:", Ws.shape)
            st.dataframe(np_to_df(Ws, row_idx=src_tokens, col_idx=src_tokens))
            st.markdown("**해석**: 각 행의 합이 1이 되며, 값이 클수록 해당 단어에 더 집중합니다.")
            
            # A-4. 최종 출력 계산
            st.markdown("#### A-4. 최종 출력 계산: Weighted Sum")
            st.markdown("어텐션 가중치를 사용하여 Value의 가중 평균을 계산합니다.")
            
            Cs = Ws @ Vs
            
            # Context Vector 표시
            st.write("**Context Vector (최종 출력)**")
            st.write("shape:", Cs.shape)
            st.dataframe(np_to_df(Cs, row_idx=src_tokens))
            st.markdown("**해석**: 각 단어가 다른 단어들의 정보를 종합한 새로운 표현입니다.")
            
            # A-5. 시각화
            st.markdown("#### A-5. 어텐션 가중치 시각화")
            plot_heatmap(Ws, xticks=src_tokens, yticks=src_tokens, title="인코더 Self-Attention 가중치")
            st.markdown("**히트맵 해석**:")
            st.markdown("- **행**: 어텐션을 주는 단어 (Query)")
            st.markdown("- **열**: 어텐션을 받는 단어 (Key)")
            st.markdown("- **색상**: 밝을수록 높은 어텐션 가중치")
            st.markdown("- **대각선**: 자기 자신에게 주는 어텐션 (보통 높음)")
            
        elif attention_type == "디코더 Masked Self-Attention":
            st.markdown("### ⛔ 디코더 마스크드 Self-Attention (타깃→타깃)")
            st.markdown("**목적**: 영어 문장을 생성할 때, 현재 단어가 이전에 생성된 단어들만 참고하도록 하여 정답을 미리 엿보지 못하게 함")
            st.markdown("**특징**: 미래 단어 정보를 차단 (단방향)")

            # C-1. Query, Key, Value 생성 (타깃에서)
            st.markdown("#### C-1. Query, Key, Value 생성 (타깃 임베딩 사용)")
            Qd_self = linear_projection(tgt_E, Wq)
            Kd_self = linear_projection(tgt_E, Wk)
            Vd_self = linear_projection(tgt_E, Wv)
            
            # Query, Key, Value 표시
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Query (타깃)**")
                st.write("shape:", Qd_self.shape)
                st.dataframe(np_to_df(Qd_self, row_idx=tgt_tokens))
            with col2:
                st.markdown("**Key (타깃)**")
                st.write("shape:", Kd_self.shape)
                st.dataframe(np_to_df(Kd_self, row_idx=tgt_tokens))
            with col3:
                st.markdown("**Value (타깃)**")
                st.write("shape:", Vd_self.shape)
                st.dataframe(np_to_df(Vd_self, row_idx=tgt_tokens))
            
            # C-2. 어텐션 스코어 계산
            st.markdown("#### C-2. 어텐션 스코어 계산")
            dk_self = Qd_self.shape[-1]
            Sd_self = Qd_self @ Kd_self.T / np.sqrt(dk_self)
            
            # Score Matrix 표시
            st.write("**Score Matrix (마스킹 전)**")
            st.dataframe(np_to_df(Sd_self, row_idx=tgt_tokens, col_idx=tgt_tokens))

            # C-3. 마스킹 적용
            st.markdown("#### C-3. 마스킹(Masking) 적용")
            st.markdown("현재 단어가 미래의 단어 정보를 참고하지 못하도록, 어텐션 스코어의 일부를 아주 작은 값(-무한대)으로 만듭니다.")
            
            mask = np.triu(np.ones_like(Sd_self), k=1).astype(bool)
            Sd_self_masked = np.where(mask, -1e9, Sd_self)

            # Score Matrix (마스킹 후) 표시
            st.write("**Score Matrix (마스킹 후)**")
            st.markdown("대각선 위쪽(미래 시점)의 값들이 `-1e9`로 변경된 것을 확인하세요.")
            st.dataframe(np_to_df(Sd_self_masked, row_idx=tgt_tokens, col_idx=tgt_tokens))
            
            # C-4. Softmax 적용 및 최종 출력 계산
            st.markdown("#### C-4. Softmax 및 최종 출력")
            st.markdown("마스킹된 스코어에 Softmax를 적용하면, 미래 단어에 대한 어텐션 가중치는 0이 됩니다.")
            Wd_self = softmax(Sd_self_masked, axis=-1)
            Cd_self = Wd_self @ Vd_self
            
            # Masked Attention Weights 표시
            st.write("**Masked Attention Weights**")
            st.dataframe(np_to_df(Wd_self, row_idx=tgt_tokens, col_idx=tgt_tokens))
            
            # C-5. 시각화
            st.markdown("#### C-5. 마스크드 어텐션 가중치 시각화")
            plot_heatmap(Wd_self, xticks=tgt_tokens, yticks=tgt_tokens, title="디코더 마스크드 Self-Attention")
            st.markdown("**히트맵 해석**: 각 행(단어)은 자기 자신과 그 이전 단어들에게만 어텐션을 주고, 미래 단어(오른쪽)는 전혀 참고하지 않습니다 (검은색).")
            
        elif attention_type == "크로스 Attention":
            st.markdown("### 🔗 인코더–디코더 Cross-Attention (타깃→소스)")
            st.markdown("**목적**: 영어 단어를 생성할 때 어떤 한국어 단어를 참고할지 결정")
            st.markdown("**특징**: 타깃에서 소스로의 정보 흐름")
            
            # B-1. Query 생성 (타깃에서)
            st.markdown("#### B-1. Query 생성: 타깃 단어에서")
            st.markdown("영어 단어들을 Query로 변환합니다.")
            
            Qd = linear_projection(tgt_E, Wq)
            
            # Query (타깃) 표시
            st.write("**Query (타깃)**")
            st.write("shape:", Qd.shape)
            st.dataframe(np_to_df(Qd, row_idx=tgt_tokens))
            st.markdown("**해석**: 영어 단어들이 '무엇을 찾고 있는지'를 나타냅니다.")
            
            # B-2. 어텐션 스코어 계산 (타깃 Query vs 소스 Key)
            st.markdown("#### B-2. 어텐션 스코어 계산: 타깃 Query vs 소스 Key")
            st.markdown("각 영어 단어가 한국어 단어들과 얼마나 관련있는지 계산합니다.")
            
            Ks = linear_projection(src_E, Wk)
            Vs = linear_projection(src_E, Wv)
            dk = Qd.shape[-1]
            Sd = Qd @ Ks.T / np.sqrt(dk)
            
            # Score Matrix (타깃 vs 소스) 표시
            st.write("**Score Matrix (타깃 vs 소스)**")
            st.write("shape:", Sd.shape)
            st.dataframe(np_to_df(Sd, row_idx=tgt_tokens, col_idx=src_tokens))
            st.markdown("**해석**: 영어 단어가 한국어 단어를 얼마나 참고할지 결정합니다.")
            
            # B-3. Softmax 적용
            st.markdown("#### B-3. Softmax 적용: 확률 분포로 변환")
            st.markdown("스코어를 확률 분포로 변환합니다.")
            
            Wd = softmax(Sd, axis=-1)
            
            # Softmax Weights 표시
            st.write("**Softmax Weights**")
            st.write("shape:", Wd.shape)
            st.dataframe(np_to_df(Wd, row_idx=tgt_tokens, col_idx=src_tokens))
            st.markdown("**해석**: 각 영어 단어가 한국어 단어들에 주는 어텐션 가중치입니다.")
            
            # B-4. 최종 출력 계산 (소스 Value 사용)
            st.markdown("#### B-4. 최종 출력 계산: 소스 Value 사용")
            st.markdown("어텐션 가중치를 사용하여 소스 정보를 종합합니다.")
            
            Cd = Wd @ Vs
            
            # Context Vector (최종 출력) 표시
            st.write("**Context Vector (최종 출력)**")
            st.write("shape:", Cd.shape)
            st.dataframe(np_to_df(Cd, row_idx=tgt_tokens))
            st.markdown("**해석**: 영어 단어가 한국어 문장의 정보를 종합한 표현입니다.")
            
            # B-5. 시각화
            st.markdown("#### B-5. 크로스 어텐션 가중치 시각화")
            plot_heatmap(Wd, xticks=src_tokens, yticks=tgt_tokens, title="크로스 어텐션 가중치 (타깃 행 / 소스 열)")
            st.markdown("**히트맵 해석**:")
            st.markdown("- **행**: 영어 단어 (타깃)")
            st.markdown("- **열**: 한국어 단어 (소스)")
            st.markdown("- **색상**: 밝을수록 해당 영어 단어가 한국어 단어를 더 참고")
            st.markdown("- **예시**: 'I'가 '나는'을 참고, 'ate'가 '먹었어'를 참고")
        


# --------------------------------------------------
# 탭 3: 소프트맥스 실험실
# --------------------------------------------------
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
    render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E)
        ax.legend()
        st.pyplot(fig)

# --------------------------------------------------
# 탭 8: PyTorch 구현
# --------------------------------------------------
with tabs[7]:
    st.subheader("PyTorch로 스케일드닷프로덕트 & 멀티헤드")
    
    if analyze:
        # PyTorch 텐서로 변환
        src_E_torch = torch.tensor(src_E, dtype=torch.float32)
        tgt_E_torch = torch.tensor(tgt_E, dtype=torch.float32)
        Wq_torch = torch.tensor(Wq, dtype=torch.float32)
        Wk_torch = torch.tensor(Wk, dtype=torch.float32)
        Wv_torch = torch.tensor(Wv, dtype=torch.float32)
        
        # PyTorch로 어텐션 계산
        Q_torch = torch.matmul(src_E_torch, Wq_torch)
        K_torch = torch.matmul(src_E_torch, Wk_torch)
        V_torch = torch.matmul(src_E_torch, Wv_torch)
        
        # Scaled Dot-Product Attention
        d_k = Q_torch.size(-1)
        scores_torch = torch.matmul(Q_torch, K_torch.transpose(-2, -1)) / math.sqrt(d_k)
        weights_torch = torch.softmax(scores_torch, dim=-1)
        context_torch = torch.matmul(weights_torch, V_torch)
        
        st.write("PyTorch 어텐션 결과 shape:", context_torch.shape)
        st.dataframe(np_to_df(context_torch.detach().numpy(), row_idx=src_tokens))

# --------------------------------------------------
# 탭 9: AI 챗봇 (Gemini AI)
# --------------------------------------------------
with tabs[8]:
    st.subheader("🤖 Gemini AI 챗봇과 대화해보세요!")
    
    # Gemini API 상태 확인
    if gemini_model:
        st.success("✅ Gemini AI가 연결되었습니다!")
        
        if analyze:
            # 분석 결과를 컨텍스트로 제공
            analysis_context = f"""
            현재 분석 중인 데이터:
            - 소스 토큰 (한국어): {src_tokens}
            - 타깃 토큰 (영어): {tgt_tokens}
            - 어텐션 분석이 완료되었습니다.
            """
            
            st.markdown("### 📊 분석 결과 기반 질문")
            
            # 미리 정의된 질문들
            predefined_questions = [
                "어텐션 메커니즘이란 무엇인가요?",
                "Q, K, V의 역할을 설명해주세요",
                "멀티헤드 어텐션의 장점은 무엇인가요?",
                "마스킹이 필요한 이유는 무엇인가요?",
                "현재 분석된 문장의 어텐션 패턴을 해석해주세요"
            ]
            
            st.markdown("**💡 추천 질문들:**")
            for i, question in enumerate(predefined_questions):
                if st.button(f"Q{i+1}: {question}", key=f"pre_q_{i}"):
                    with st.spinner("🤔 Gemini AI가 생각하고 있습니다..."):
                        try:
                            prompt = f"""
                            {analysis_context}
                            
                            질문: {question}
                            
                            어텐션 메커니즘에 대한 교육적이고 이해하기 쉬운 답변을 한국어로 제공해주세요.
                            """
                            
                            response = gemini_model.generate_content(prompt)
                            st.markdown(f"**🤖 Gemini AI 답변:**")
                            st.markdown(response.text)
                            
                        except Exception as e:
                            st.error(f"❌ Gemini AI 응답 생성 실패: {str(e)}")
            
            st.markdown("---")
        
        # 자유 질문
        st.markdown("### 💬 자유 질문")
        user_input = st.text_area(
            "어텐션 메커니즘에 대해 자유롭게 질문해보세요:",
            placeholder="예: Transformer와 RNN의 차이점은 무엇인가요?",
            height=100
        )
        
        if st.button("🤖 Gemini AI에게 질문하기"):
            if user_input.strip():
                with st.spinner("🤔 Gemini AI가 생각하고 있습니다..."):
                    try:
                        # 컨텍스트 정보 추가
                        context = ""
                        if analyze:
                            context = f"""
                            현재 분석 중인 데이터:
                            - 소스 토큰 (한국어): {src_tokens}
                            - 타깃 토큰 (영어): {tgt_tokens}
                            
                            """
                        
                        prompt = f"""
                        {context}
                        
                        질문: {user_input}
                        
                        어텐션 메커니즘과 관련된 질문에 대해 교육적이고 이해하기 쉬운 답변을 한국어로 제공해주세요.
                        가능하면 구체적인 예시와 함께 설명해주세요.
                        """
                        
                        response = gemini_model.generate_content(prompt)
                        st.markdown(f"**🤖 Gemini AI 답변:**")
                        st.markdown(response.text)
                        
                        # 대화 히스토리 저장
                        if 'chat_history' not in st.session_state:
                            st.session_state.chat_history = []
                        
                        st.session_state.chat_history.append({
                            'question': user_input,
                            'answer': response.text,
                            'timestamp': 'now'
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Gemini AI 응답 생성 실패: {str(e)}")
                        st.info("💡 API 키를 확인하거나 잠시 후 다시 시도해보세요.")
            else:
                st.warning("질문을 입력해주세요.")
        
        # 대화 히스토리 표시
        if 'chat_history' in st.session_state and st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 📝 대화 히스토리")
            
            for i, chat in enumerate(st.session_state.chat_history[-5:]):  # 최근 5개만 표시
                with st.expander(f"질문 {i+1}: {chat['question'][:50]}..."):
                    st.markdown(f"**질문:** {chat['question']}")
                    st.markdown(f"**답변:** {chat['answer']}")
            
            if st.button("🗑️ 대화 히스토리 삭제"):
                st.session_state.chat_history = []
                st.rerun()
    
    else:
        st.warning("⚠️ Gemini AI를 사용하려면 API 키를 설정해주세요.")
        st.info("""
        **API 키 설정 방법:**
        1. [Google AI Studio](https://aistudio.google.com/app/apikey)에서 API 키 발급
        2. 사이드바에 API 키 입력
        3. 또는 환경 변수 `GOOGLE_API_KEY`에 설정
        
        **또는 기존 키워드 기반 챗봇 사용:**
        """)
        
        # 기존 키워드 기반 챗봇 (API 키가 없을 때)
        if analyze:
            st.markdown("### 📊 분석 결과 기반 질문")
            
            Qd = linear_projection(tgt_E, Wq)
            Ks = linear_projection(src_E, Wk)
            Vs = linear_projection(src_E, Wv)
            _, Wd, _ = scaled_dot_product_attention(Qd, Ks, Vs)
            
            if len(tgt_tokens) > 0 and len(src_tokens) > 0:
                q1 = f"'{tgt_tokens[0]}' 단어가 가장 주목한 원본 단어는 무엇인가요?"
                if st.button(q1, key="fallback_q1"):
                    focused_idx = np.argmax(Wd[0])
                    focused_word = src_tokens[focused_idx]
                    st.write(f"**답변:** '{focused_word}' 단어입니다. (어텐션 스코어: {Wd[0, focused_idx]:.3f})")
                
                q2 = "가장 높은 어텐션 스코어를 가진 단어 쌍은 무엇인가요?"
                if st.button(q2, key="fallback_q2"):
                    max_idx = np.unravel_index(np.argmax(Wd), Wd.shape)
                    max_score = Wd[max_idx]
                    tgt_word = tgt_tokens[max_idx[0]]
                    src_word = src_tokens[max_idx[1]]
                    st.write(f"**답변:** '{tgt_word}' → '{src_word}' (어텐션 스코어: {max_score:.3f})")
        
        st.markdown("### 💬 키워드 기반 질문")
        user_input = st.text_input("키워드로 질문해보세요:", placeholder="예: attention, transformer, masking...")
        if st.button("질문하기", key="fallback_btn"):
            responses = {
                "attention": "어텐션 메커니즘은 입력 시퀀스의 특정 부분에 집중하여 출력을 생성하는 방법입니다.",
                "transformer": "Transformer는 어텐션 메커니즘을 기반으로 한 신경망 아키텍처입니다.",
                "self-attention": "Self-attention은 같은 시퀀스 내의 다른 위치들을 참조하는 어텐션입니다.",
                "cross-attention": "Cross-attention은 서로 다른 시퀀스 간의 어텐션입니다.",
                "멀티헤드": "멀티헤드 어텐션은 여러 개의 어텐션을 병렬로 수행하여 다양한 관계를 동시에 파악하는 방법입니다.",
                "마스킹": "마스킹은 디코더에서 미래 토큰을 보지 못하게 하여 학습 시 정답을 미리 엿보지 못하게 하는 기법입니다.",
                "q": "Query(질문)는 현재 단어가 다른 단어들과 어떤 관계를 맺어야 할지 알아보기 위해 던지는 질문입니다.",
                "k": "Key(키)는 다른 단어들이 '나 이런 정보를 가지고 있어!'라고 알려주는 역할을 합니다.",
                "v": "Value(값)는 실제 내용으로, Query와 가장 잘 맞는 Key에 연결된 정보를 가져옵니다."
            }
            
            response = "죄송합니다. 질문에 대한 답변을 찾을 수 없습니다. 'attention', 'transformer', 'self-attention', 'cross-attention', '멀티헤드', '마스킹', 'q', 'k', 'v' 등에 대해 질문해보세요."
            for keyword, resp in responses.items():
                if keyword.lower() in user_input.lower():
                    response = resp
                    break
            
            st.write("**AI 응답:**", response)
            st.info("💡 이는 키워드 기반 응답입니다. Gemini AI를 사용하면 더 정교한 답변을 받을 수 있습니다.")

# --------------------------------------------------
# 탭 10: 퀴즈
# --------------------------------------------------
with tabs[9]:
    st.subheader("📝 어텐션 메커니즘 퀴즈")
    st.markdown("학습한 내용을 확인해보세요!")
    
    # 퀴즈 1: Q, K, V의 역할
    st.markdown("### 🎯 퀴즈 1: Q, K, V의 역할")
    st.markdown("**질문**: 어텐션에서 Query(Q)의 역할은 무엇인가요?")
    
    q1_answer = st.radio(
        "정답을 선택하세요:",
        [
            "다른 단어들이 가지고 있는 정보의 특징을 나타냄",
            "현재 단어가 다른 단어들과 어떤 관계를 맺어야 할지 알아보기 위해 던지는 질문",
            "각 단어가 실제로 담고 있는 의미 정보",
            "어텐션 가중치를 계산하는 데 사용되는 스케일링 팩터"
        ],
        key="q1"
    )
    
    if st.button("정답 확인", key="check1"):
        if q1_answer == "현재 단어가 다른 단어들과 어떤 관계를 맺어야 할지 알아보기 위해 던지는 질문":
            st.success("🎉 정답입니다! Query는 '무엇을 찾고 있는가?'를 나타내는 질문입니다.")
        else:
            st.error("❌ 틀렸습니다. Query는 현재 단어가 다른 단어들과 어떤 관계를 맺어야 할지 알아보기 위해 던지는 질문입니다.")
    
    st.markdown("---")
    
    # 퀴즈 2: 마스킹의 목적
    st.markdown("### 🎯 퀴즈 2: 마스킹의 목적")
    st.markdown("**질문**: 디코더에서 마스킹을 사용하는 이유는 무엇인가요?")
    
    q2_answer = st.radio(
        "정답을 선택하세요:",
        [
            "계산 속도를 높이기 위해",
            "메모리 사용량을 줄이기 위해",
            "미래 토큰 정보를 참고하지 못하게 하여 정답을 미리 엿보지 못하게 하기 위해",
            "어텐션 가중치의 분산을 줄이기 위해"
        ],
        key="q2"
    )
    
    if st.button("정답 확인", key="check2"):
        if q2_answer == "미래 토큰 정보를 참고하지 못하게 하여 정답을 미리 엿보지 못하게 하기 위해":
            st.success("🎉 정답입니다! 마스킹은 학습 시 정답을 미리 보는 것을 방지합니다.")
        else:
            st.error("❌ 틀렸습니다. 마스킹은 미래 토큰 정보를 차단하여 정답을 미리 엿보지 못하게 합니다.")
    
    st.markdown("---")
    
    # 퀴즈 3: 멀티헤드 어텐션
    st.markdown("### 🎯 퀴즈 3: 멀티헤드 어텐션")
    st.markdown("**질문**: 멀티헤드 어텐션의 장점은 무엇인가요?")
    
    q3_answer = st.radio(
        "정답을 선택하세요:",
        [
            "계산 복잡도를 줄임",
            "여러 개의 '헤드'를 두어 다양한 관점에서 정보를 종합할 수 있음",
            "메모리 사용량을 줄임",
            "학습률을 자동으로 조정함"
        ],
        key="q3"
    )
    
    if st.button("정답 확인", key="check3"):
        if q3_answer == "여러 개의 '헤드'를 두어 다양한 관점에서 정보를 종합할 수 있음":
            st.success("🎉 정답입니다! 멀티헤드 어텐션은 다양한 관점에서 정보를 종합할 수 있습니다.")
        else:
            st.error("❌ 틀렸습니다. 멀티헤드 어텐션의 핵심은 여러 헤드가 다양한 관점에서 정보를 종합하는 것입니다.")
    
    # 점수 계산
    st.markdown("---")
    st.markdown("### 📊 퀴즈 결과")
    
    # 간단한 점수 계산 (실제로는 더 정교하게 구현 가능)
    correct_answers = 0
    if 'q1' in st.session_state and st.session_state.q1 == "현재 단어가 다른 단어들과 어떤 관계를 맺어야 할지 알아보기 위해 던지는 질문":
        correct_answers += 1
    if 'q2' in st.session_state and st.session_state.q2 == "미래 토큰 정보를 참고하지 못하게 하여 정답을 미리 엿보지 못하게 하기 위해":
        correct_answers += 1
    if 'q3' in st.session_state and st.session_state.q3 == "여러 개의 '헤드'를 두어 다양한 관점에서 정보를 종합할 수 있음":
        correct_answers += 1
    
    st.metric("정답 수", f"{correct_answers}/3")
    
    if correct_answers == 3:
        st.success("🎉 완벽합니다! 어텐션 메커니즘을 잘 이해하고 있습니다.")
    elif correct_answers >= 2:
        st.info("👍 잘하고 있습니다! 조금만 더 학습하면 됩니다.")
    else:
        st.warning("📚 더 많은 학습이 필요합니다. 위의 학습 가이드를 다시 살펴보세요.")

# --------------------------------------------------
# 탭 11: 용어사전
# --------------------------------------------------
with tabs[10]:
    st.subheader("📚 어텐션 메커니즘 용어사전")
    st.markdown("핵심 개념들을 정리한 용어사전입니다.")
    
    # 검색 기능
    search_term = st.text_input("🔍 용어 검색:", placeholder="예: attention, masking, transformer...")
    
    # 용어사전 데이터
    glossary = {
        "attention": {
            "한글": "어텐션",
            "정의": "입력의 특정 부분에 집중하여 출력을 생성하는 메커니즘",
            "설명": "모든 입력을 동시에 고려하되, 중요한 부분에 더 집중하는 방식으로 작동합니다."
        },
        "query": {
            "한글": "쿼리",
            "정의": "현재 단어가 다른 단어들과 어떤 관계를 맺어야 할지 알아보기 위해 던지는 질문",
            "설명": "도서관에서 검색어를 입력하는 것과 비슷한 개념입니다."
        },
        "key": {
            "한글": "키",
            "정의": "다른 단어들이 가지고 있는 정보의 특징",
            "설명": "도서관의 책 제목이나 키워드와 같은 역할을 합니다."
        },
        "value": {
            "한글": "값",
            "정의": "각 단어가 실제로 담고 있는 의미 정보",
            "설명": "도서관의 책 내용과 같은 실제 정보입니다."
        },
        "self-attention": {
            "한글": "셀프 어텐션",
            "정의": "같은 시퀀스 내의 다른 위치들을 참조하는 어텐션",
            "설명": "문장 내에서 단어들 간의 관계를 학습합니다."
        },
        "cross-attention": {
            "한글": "크로스 어텐션",
            "정의": "서로 다른 시퀀스 간의 어텐션",
            "설명": "번역에서 소스 언어와 타깃 언어 간의 관계를 학습합니다."
        },
        "masking": {
            "한글": "마스킹",
            "정의": "특정 위치의 정보를 차단하는 기법",
            "설명": "디코더에서 미래 토큰을 보지 못하게 하여 인과성을 보장합니다."
        },
        "multi-head": {
            "한글": "멀티헤드",
            "정의": "여러 개의 어텐션을 병렬로 수행하는 구조",
            "설명": "다양한 관점에서 정보를 종합할 수 있습니다."
        },
        "transformer": {
            "한글": "트랜스포머",
            "정의": "어텐션 메커니즘을 기반으로 한 신경망 아키텍처",
            "설명": "RNN의 순차적 처리 한계를 극복한 병렬 처리 가능한 구조입니다."
        },
        "positional encoding": {
            "한글": "위치 인코딩",
            "정의": "토큰의 위치 정보를 임베딩에 추가하는 기법",
            "설명": "어텐션은 위치 정보가 없으므로 별도로 위치 정보를 제공해야 합니다."
        }
    }
    
    # 검색 결과 표시
    if search_term:
        search_term = search_term.lower()
        found_terms = {k: v for k, v in glossary.items() if search_term in k or search_term in v["한글"]}
        
        if found_terms:
            st.markdown(f"**검색 결과: '{search_term}'**")
            for term, info in found_terms.items():
                with st.expander(f"**{term}** ({info['한글']})"):
                    st.markdown(f"**정의**: {info['정의']}")
                    st.markdown(f"**설명**: {info['설명']}")
        else:
            st.warning(f"'{search_term}'에 대한 용어를 찾을 수 없습니다.")
    
    # 전체 용어사전 표시
    st.markdown("### 📖 전체 용어사전")
    for term, info in glossary.items():
        with st.expander(f"**{term}** ({info['한글']})"):
            st.markdown(f"**정의**: {info['정의']}")
            st.markdown(f"**설명**: {info['설명']}")
    
    st.markdown("---")
    st.markdown("💡 **팁**: 위의 학습 가이드와 함께 용어사전을 참고하면 더욱 효과적으로 학습할 수 있습니다.")