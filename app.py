
# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
from dataclasses import dataclass
import math

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

tabs = st.tabs(["🧭 학습 가이드(초보자용)", "🔬 단계별 어텐션(실험실)", "🧩 멀티헤드 시각화", "⛔ 마스킹 & 인과성", "어텐션 지도", "임베딩 분석", "PyTorch 구현", "AI 챗봇", "📝 퀴즈", "📚 용어사전"])

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
    st.subheader("🎯 어텐션 메커니즘 학습 가이드")
    st.markdown("""
    어텐션 메커니즘을 단계별로 학습해보세요. 각 단계를 클릭하면 해당 내용을 자세히 볼 수 있습니다.
    """)
    
    # 단계별 학습 가이드
    step = st.selectbox("학습 단계 선택", 
                        ["0. 개요", "1. 토큰화 & 임베딩", "2. 선형사영(Q,K,V)", 
                         "3. 유사도(QKᵀ)", "4. 스케일링(/√dₖ)", "5. 소프트맥스(가중치)", "6. 가중합(컨텍스트)"])
    
    if step == "0. 개요":
        st.markdown("""
        ### 🎯 어텐션 메커니즘이란?
        
        **어텐션(Attention)**은 딥러닝에서 입력의 특정 부분에 집중하여 출력을 생성하는 메커니즘입니다.
        
        **핵심 아이디어**: 
        - 입력 시퀀스의 모든 위치를 동시에 고려
        - 각 위치의 중요도를 동적으로 계산
        - 중요한 정보에 더 집중하여 처리
        
        **트랜스포머에서의 역할**:
        1. **Self-Attention**: 같은 시퀀스 내의 관계 학습
        2. **Cross-Attention**: 서로 다른 시퀀스 간의 관계 학습
        3. **Masked Attention**: 미래 정보를 보지 못하게 하는 제약
        """)
        
    elif step == "1. 토큰화 & 임베딩":
        st.markdown("""
        ### 📝 1단계: 토큰화 & 임베딩
        
        **토큰화(Tokenization)**:
        - 텍스트를 의미 있는 단위(토큰)로 분리
        - 예: "나는 밥을 먹었어" → ["나는", "밥을", "먹었어"]
        
        **임베딩(Embedding)**:
        - 각 토큰을 고차원 벡터로 변환
        - 의미적 정보를 수치로 표현
        - 위치 정보도 함께 인코딩 (Positional Encoding)
        """)
        
        if analyze:
            st.write("**현재 예시**:")
            col1, col2 = st.columns(2)
            with col1:
                st.write("소스 토큰:", src_tokens)
                st.write("소스 임베딩 shape:", src_E.shape)
            with col2:
                st.write("타깃 토큰:", tgt_tokens)
                st.write("타깃 임베딩 shape:", tgt_E.shape)
    
    elif step == "2. 선형사영(Q,K,V)":
        st.markdown("""
        ### 🔄 2단계: 선형사영으로 Q, K, V 생성
        
        **Q(Query), K(Key), V(Value)의 역할**:
        
        - **Query (Q)**: "무엇을 찾고 있는가?" - 현재 단어가 다른 단어들과 어떤 관계를 맺어야 할지
        - **Key (K)**: "무엇을 제공할 수 있는가?" - 다른 단어들이 가지고 있는 정보의 특징
        - **Value (V)**: "실제 내용은 무엇인가?" - 각 단어가 실제로 담고 있는 의미 정보
        
        **수식**: `Q = XW_Q`, `K = XW_K`, `V = XW_V`
        """)
        
        if analyze:
            Qs = linear_projection(src_E, Wq)
            Ks = linear_projection(src_E, Wk)
            Vs = linear_projection(src_E, Wv)
            
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
    
    elif step == "3. 유사도(QKᵀ)":
        st.markdown("""
        ### 🔍 3단계: 유사도 계산 (QKᵀ)
        
        **목적**: 각 Query와 모든 Key 간의 유사도를 계산
        
        **수식**: `Scores = QK^T`
        
        **의미**: 
        - 값이 클수록 해당 Query-Key 쌍이 더 관련성이 높음
        - 어텐션을 줄 단어를 결정하는 핵심 단계
        """)
        
        if analyze:
            Qs = linear_projection(src_E, Wq)
            Ks = linear_projection(src_E, Wk)
            Ss = Qs @ Ks.T
            
            st.write("**Score Matrix (QKᵀ)**")
            st.write("shape:", Ss.shape)
            st.dataframe(np_to_df(Ss, row_idx=src_tokens, col_idx=src_tokens))
            
            st.markdown("**해석**:")
            st.markdown("- 행: Query (어텐션을 주는 단어)")
            st.markdown("- 열: Key (어텐션을 받는 단어)")
            st.markdown("- 값이 클수록 더 관련성이 높음")
    
    elif step == "4. 스케일링(/√dₖ)":
        st.markdown("""
        ### 📏 4단계: 스케일링 (/√dₖ)
        
        **목적**: 어텐션 스코어를 안정적으로 만들기
        
        **수식**: `Scores = QK^T / √d_k`
        
        **왜 필요한가?**:
        - d_k가 클 때 QK^T 값이 너무 커져서 softmax에서 기울기 소실 발생
        - √d_k로 나누어 분산을 1로 정규화
        - 학습 안정성 향상
        """)
        
        if analyze:
            Qs = linear_projection(src_E, Wq)
            Ks = linear_projection(src_E, Wk)
            dk = Qs.shape[-1]
            
            # 스케일링 전후 비교
            Ss_raw = Qs @ Ks.T
            Ss_scaled = Ss_raw / np.sqrt(dk)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**스케일링 전**")
                st.write("표준편차:", np.std(Ss_raw))
                st.dataframe(np_to_df(Ss_raw, row_idx=src_tokens, col_idx=src_tokens))
            with col2:
                st.markdown("**스케일링 후**")
                st.write("표준편차:", np.std(Ss_scaled))
                st.dataframe(np_to_df(Ss_scaled, row_idx=src_tokens, col_idx=src_tokens))
    
    elif step == "5. 소프트맥스(가중치)":
        st.markdown("""
        ### 🎲 5단계: Softmax로 가중치 생성
        
        **목적**: 스코어를 확률 분포로 변환
        
        **수식**: `Weights = softmax(Scores)`
        
        **특징**:
        - 각 행의 합이 1이 됨
        - 값이 클수록 더 높은 어텐션 가중치
        - 수치 안정성을 위해 `softmax(x - max(x))` 사용
        """)
        
        if analyze:
            Qs = linear_projection(src_E, Wq)
            Ks = linear_projection(src_E, Wk)
            dk = Qs.shape[-1]
            Ss = Qs @ Ks.T / np.sqrt(dk)
            Ws = softmax(Ss, axis=-1)
            
            st.write("**Softmax Weights**")
            st.write("shape:", Ws.shape)
            st.dataframe(np_to_df(Ws, row_idx=src_tokens, col_idx=src_tokens))
            
            # 수치 안정성 데모
            st.markdown("**수치 안정성 데모**:")
            st.markdown("`softmax(x - max(x))`를 사용하는 이유:")
            
            # 큰 값으로 테스트
            test_scores = np.array([1000, 1001, 1002])
            st.write("테스트 스코어:", test_scores)
            
            # 일반적인 softmax (오버플로우 위험)
            try:
                exp_scores = np.exp(test_scores)
                softmax_normal = exp_scores / np.sum(exp_scores)
                st.write("일반 softmax:", softmax_normal)
            except:
                st.error("오버플로우 발생!")
            
            # 안정적인 softmax
            stable_scores = test_scores - np.max(test_scores)
            exp_stable = np.exp(stable_scores)
            softmax_stable = exp_stable / np.sum(exp_stable)
            st.write("안정적 softmax:", softmax_stable)
    
    elif step == "6. 가중합(컨텍스트)":
        st.markdown("""
        ### 🎯 6단계: 가중합으로 컨텍스트 벡터 생성
        
        **목적**: 어텐션 가중치를 사용하여 Value의 가중 평균 계산
        
        **수식**: `Context = Weights × V`
        
        **결과**: 
        - 각 단어가 다른 단어들의 정보를 종합한 새로운 표현
        - 어텐션 메커니즘의 최종 출력
        - 다음 레이어의 입력으로 사용
        """)
        
        if analyze:
            Qs = linear_projection(src_E, Wq)
            Ks = linear_projection(src_E, Wk)
            Vs = linear_projection(src_E, Wv)
            dk = Qs.shape[-1]
            Ss = Qs @ Ks.T / np.sqrt(dk)
            Ws = softmax(Ss, axis=-1)
            Cs = Ws @ Vs
            
            st.write("**Context Vector (최종 출력)**")
            st.write("shape:", Cs.shape)
            st.dataframe(np_to_df(Cs, row_idx=src_tokens))
            
            st.markdown("**해석**:")
            st.markdown("- 각 행: 해당 단어가 다른 단어들의 정보를 종합한 표현")
            st.markdown("- 원본 임베딩과 다른 새로운 벡터")
            st.markdown("- 어텐션 메커니즘의 핵심 결과물")

# --------------------------------------------------
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
    
    if show_formula:
        st.markdown("**공식**")
        st.latex(r"Q = XW_Q,\ K = XW_K,\ V = XW_V")
        st.latex(r"\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V")
        st.caption("※ 여기서는 교육용으로 소스/타깃 임베딩에 같은 차원/가중치를 사용합니다.")
    
    st.markdown("**토큰**")
    c1, c2 = st.columns(2)
    with c1:
        st.write("소스 토큰:", src_tokens)
    with c2:
        st.write("타깃 토큰:", tgt_tokens)

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
            
            if show_steps:
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
            
            if show_steps:
                st.write("**Score Matrix (QKᵀ/√d)**")
                st.write("shape:", Ss.shape)
                st.dataframe(np_to_df(Ss, row_idx=src_tokens, col_idx=src_tokens))
                st.markdown("**해석**: 값이 클수록 해당 단어 쌍이 더 관련성이 높습니다.")
            
            # A-3. Softmax 적용
            st.markdown("#### A-3. Softmax 적용: 확률 분포로 변환")
            st.markdown("스코어를 확률 분포로 변환하여 어텐션 가중치를 만듭니다.")
            
            Ws = softmax(Ss, axis=-1)
            
            if show_steps:
                st.write("**Softmax Weights**")
                st.write("shape:", Ws.shape)
                st.dataframe(np_to_df(Ws, row_idx=src_tokens, col_idx=src_tokens))
                st.markdown("**해석**: 각 행의 합이 1이 되며, 값이 클수록 해당 단어에 더 집중합니다.")
            
            # A-4. 최종 출력 계산
            st.markdown("#### A-4. 최종 출력 계산: Weighted Sum")
            st.markdown("어텐션 가중치를 사용하여 Value의 가중 평균을 계산합니다.")
            
            Cs = Ws @ Vs
            
            if show_steps:
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
            
            if show_steps:
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
            
            if show_steps:
                st.write("**Score Matrix (마스킹 전)**")
                st.dataframe(np_to_df(Sd_self, row_idx=tgt_tokens, col_idx=tgt_tokens))

            # C-3. 마스킹 적용
            st.markdown("#### C-3. 마스킹(Masking) 적용")
            st.markdown("현재 단어가 미래의 단어 정보를 참고하지 못하도록, 어텐션 스코어의 일부를 아주 작은 값(-무한대)으로 만듭니다.")
            
            mask = np.triu(np.ones_like(Sd_self), k=1).astype(bool)
            Sd_self_masked = np.where(mask, -1e9, Sd_self)

            if show_steps:
                st.write("**Score Matrix (마스킹 후)**")
                st.markdown("대각선 위쪽(미래 시점)의 값들이 `-1e9`로 변경된 것을 확인하세요.")
                st.dataframe(np_to_df(Sd_self_masked, row_idx=tgt_tokens, col_idx=tgt_tokens))
            
            # C-4. Softmax 적용 및 최종 출력 계산
            st.markdown("#### C-4. Softmax 및 최종 출력")
            st.markdown("마스킹된 스코어에 Softmax를 적용하면, 미래 단어에 대한 어텐션 가중치는 0이 됩니다.")
            Wd_self = softmax(Sd_self_masked, axis=-1)
            Cd_self = Wd_self @ Vd_self
            
            if show_steps:
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
            
            if show_steps:
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
            
            if show_steps:
                st.write("**Score Matrix (타깃 vs 소스)**")
                st.write("shape:", Sd.shape)
                st.dataframe(np_to_df(Sd, row_idx=tgt_tokens, col_idx=src_tokens))
                st.markdown("**해석**: 영어 단어가 한국어 단어를 얼마나 참고할지 결정합니다.")
            
            # B-3. Softmax 적용
            st.markdown("#### B-3. Softmax 적용: 확률 분포로 변환")
            st.markdown("스코어를 확률 분포로 변환합니다.")
            
            Wd = softmax(Sd, axis=-1)
            
            if show_steps:
                st.write("**Softmax Weights**")
                st.write("shape:", Wd.shape)
                st.dataframe(np_to_df(Wd, row_idx=tgt_tokens, col_idx=src_tokens))
                st.markdown("**해석**: 각 영어 단어가 한국어 단어들에 주는 어텐션 가중치입니다.")
            
            # B-4. 최종 출력 계산 (소스 Value 사용)
            st.markdown("#### B-4. 최종 출력 계산: 소스 Value 사용")
            st.markdown("어텐션 가중치를 사용하여 소스 정보를 종합합니다.")
            
            Cd = Wd @ Vs
            
            if show_steps:
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
# 탭 3: 멀티헤드 시각화
# --------------------------------------------------
with tabs[2]:
    st.subheader("🧩 Multi-Head Attention 시각화")
    st.markdown("""
    한 번의 어텐션이 한 가지 종류의 관계만 본다면, 여러 번의 어텐션을 병렬로 수행하여 다양한 관계(예: 문법적 관계, 의미적 관계 등)를 동시에 파악할 수 있습니다. 이것이 바로 **멀티헤드 어텐션**의 핵심 아이디어입니다.
    
    **과정:**
    1. **분할(Split)**: 기존의 Q, K, V를 여러 개('헤드'의 수만큼)의 작은 조각으로 나눕니다.
    2. **병렬 어텐션(Parallel Attention)**: 각 조각(헤드)에 대해 독립적으로 Scaled Dot-Product 어텐션을 수행합니다. 각 헤드는 서로 다른 어텐션 패턴을 학습하게 됩니다.
    3. **결합(Concatenate)**: 각 헤드에서 나온 결과(Context Vector)들을 다시 하나로 합칩니다.
    4. **최종 변환(Final Projection)**: 합쳐진 벡터를 최종 출력 차원으로 변환하는 선형 레이어를 통과시킵니다.
    """)
    st.latex(r"\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O \quad \text{where head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)")
    
    if analyze:
        st.markdown("---")
        
        # 헤드 수 선택
        num_heads = st.slider("헤드 수 선택", min_value=1, max_value=8, value=4, help="분석할 헤드의 수를 선택하세요")
        
        st.markdown(f"### 🎯 멀티헤드 크로스 어텐션 예시 (Heads = {num_heads})")
        
        head_dim = dim // num_heads
        
        # 편의상, 기존 Q,K,V를 분할하여 사용
        Qd = linear_projection(tgt_E, Wq)
        Ks = linear_projection(src_E, Wk)
        Vs = linear_projection(src_E, Wv)
        
        # 1. 분할 (reshape)
        # (len, dim) -> (len, num_heads, head_dim) -> (num_heads, len, head_dim)
        Qd_heads = Qd.reshape(len(tgt_tokens), num_heads, head_dim).transpose(1, 0, 2)
        Ks_heads = Ks.reshape(len(src_tokens), num_heads, head_dim).transpose(1, 0, 2)
        Vs_heads = Vs.reshape(len(src_tokens), num_heads, head_dim).transpose(1, 0, 2)
        
        st.write(f"**1. 분할**: Q, K, V를 각각 `{num_heads}`개의 헤드로 나눕니다.")
        st.write(f" - 원본 Q shape: `{Qd.shape}` -> 헤드별 Q shape: `{Qd_heads.shape}` (num_heads, len, head_dim)")
        st.write(f" - 각 헤드의 차원: `{head_dim}` (원본 차원 `{dim}` ÷ 헤드 수 `{num_heads}`)")
        
        # 2. 병렬 어텐션 & 시각화
        st.write(f"**2. 병렬 어텐션**: 각 헤드가 독립적으로 어텐션을 계산합니다.")
        
        # 헤드별 어텐션 가중치 저장
        head_weights = []
        head_outputs = []
        
        # 헤드별 개별 플롯으로 표시
        for i in range(num_heads):
            context, weights, _ = scaled_dot_product_attention(Qd_heads[i], Ks_heads[i], Vs_heads[i])
            head_outputs.append(context)
            head_weights.append(weights)
            
            # 개별 플롯
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(weights, cmap='viridis')
            ax.set_xticks(np.arange(len(src_tokens)))
            ax.set_yticks(np.arange(len(tgt_tokens)))
            ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=10)
            ax.set_yticklabels(tgt_tokens, fontsize=10)
            ax.set_title(f"Head {i+1} Attention Weights", fontsize=12)
            ax.set_xlabel("Source Tokens (한국어)", fontsize=10)
            ax.set_ylabel("Target Tokens (영어)", fontsize=10)
            
            # 값 주석 (헤드가 많으면 생략)
            if num_heads <= 4:
                for y in range(len(tgt_tokens)):
                    for x in range(len(src_tokens)):
                        ax.text(x, y, f"{weights[y, x]:.2f}", ha="center", va="center", fontsize=8)
            
            # 컬러바 추가
            plt.colorbar(im, ax=ax, shrink=0.8)
            st.pyplot(fig)
            
            # 헤드별 통계 정보
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"Head {i+1} 평균 어텐션", f"{np.mean(weights):.4f}")
            with col2:
                st.metric(f"Head {i+1} 최대 어텐션", f"{np.max(weights):.4f}")
            with col3:
                st.metric(f"Head {i+1} 엔트로피", f"{calculate_entropy(weights):.4f}")
            
            st.markdown("---")

        # 3. 헤드 간 비교 분석
        st.markdown("### 🔍 헤드 간 비교 분석")
        
        # 모든 헤드의 어텐션 가중치를 하나의 히트맵으로 비교
        if num_heads > 1:
            # 평균 어텐션 가중치
            avg_weights = np.mean(head_weights, axis=0)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(avg_weights, cmap='viridis')
            ax.set_xticks(np.arange(len(src_tokens)))
            ax.set_yticks(np.arange(len(tgt_tokens)))
            ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=10)
            ax.set_yticklabels(tgt_tokens, fontsize=10)
            ax.set_title(f"평균 어텐션 가중치 (모든 헤드)", fontsize=12)
            ax.set_xlabel("Source Tokens (한국어)", fontsize=10)
            ax.set_ylabel("Target Tokens (영어)", fontsize=10)
            
            # 값 주석
            for y in range(len(tgt_tokens)):
                for x in range(len(src_tokens)):
                    ax.text(x, y, f"{avg_weights[y, x]:.3f}", ha="center", va="center", fontsize=9)
            
            plt.colorbar(im, ax=ax, shrink=0.8)
            st.pyplot(fig)
            
            st.markdown("**해석**: 모든 헤드의 평균을 취하면, 각 헤드가 학습한 다양한 패턴들이 종합된 결과를 볼 수 있습니다.")
        
        # 4. 결합 및 최종 변환
        st.write("**3. 결합 및 최종 변환**: 결과들을 합치고 최종 출력 벡터를 만듭니다.")
        concatenated = np.concatenate([h.transpose(1,0) for h in head_outputs], axis=-1).reshape(len(tgt_tokens), dim)
        # Wo는 시연용으로 생략
        st.write(" - Concatenated shape:", concatenated.shape)
        st.dataframe(np_to_df(concatenated, row_idx=tgt_tokens))
        
        # 5. 헤드별 특성 분석
        st.markdown("### 📊 헤드별 특성 분석")
        
        # 헤드별 엔트로피 계산
        entropies = [calculate_entropy(weights) for weights in head_weights]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(1, num_heads + 1), entropies)
        ax.set_xlabel("Head Number", fontsize=12)
        ax.set_ylabel("Entropy", fontsize=12)
        ax.set_title("각 헤드의 어텐션 엔트로피", fontsize=14)
        ax.set_xticks(range(1, num_heads + 1))
        
        # 값 표시
        for bar, entropy in zip(bars, entropies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{entropy:.3f}', ha='center', va='bottom')
        
        st.pyplot(fig)
        
        st.markdown("**엔트로피 해석**:")
        st.markdown("- **높은 엔트로피**: 해당 헤드가 여러 소스 토큰에 고르게 주목 (분산된 어텐션)")
        st.markdown("- **낮은 엔트로피**: 해당 헤드가 특정 소스 토큰에 집중 (집중된 어텐션)")
        
        # 헤드별 패턴 요약
        st.markdown("### 🎯 헤드별 패턴 요약")
        for i, weights in enumerate(head_weights):
            max_attention_idx = np.unravel_index(np.argmax(weights), weights.shape)
            max_tgt, max_src = max_attention_idx
            max_score = weights[max_attention_idx]
            
            st.markdown(f"**Head {i+1}**: '{tgt_tokens[max_tgt]}' → '{src_tokens[max_src]}' (어텐션: {max_score:.3f})")
        
        st.markdown("**해석**: 각 헤드가 서로 다른 어텐션 패턴을 보이는 것을 확인할 수 있습니다. 어떤 헤드는 단어 대 단어 관계에, 다른 헤드는 좀 더 분산된 관계에 집중합니다.")

# --------------------------------------------------
# 탭 4: 마스킹 & 인과성
# --------------------------------------------------
with tabs[3]:
    st.subheader("⛔ 마스킹 & 인과성 (Causality)")
    st.markdown("""
    디코더에서 미래 토큰을 보지 못하게 하는 마스킹의 원리와 수학적 표현을 살펴봅니다.
    """)
    
    if analyze:
        # 마스킹 매트릭스 생성
        seq_len = len(tgt_tokens)
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        
        st.markdown("### 🔒 Causal Mask Matrix")
        st.markdown("**수식**: `M_{ij} = 1[j ≤ i]` (j ≤ i일 때만 1, 그렇지 않으면 0)")
        st.markdown("**의미**: i번째 위치에서 j번째 위치를 참고할 수 있는지 여부")
        
        # 마스킹 매트릭스 시각화
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(mask.astype(int), cmap='RdYlBu_r')
        ax.set_xticks(np.arange(seq_len))
        ax.set_yticks(np.arange(seq_len))
        ax.set_xticklabels(tgt_tokens, rotation=45, ha="right")
        ax.set_yticklabels(tgt_tokens)
        ax.set_title("Causal Mask Matrix (1=참고 가능, 0=참고 불가)", fontsize=12)
        
        # 값 주석
        for i in range(seq_len):
            for j in range(seq_len):
                text = ax.text(j, i, "✓" if mask[i, j] == False else "✗", 
                             ha="center", va="center", color="black", fontsize=14)
        
        st.pyplot(fig)
        
        st.markdown("**해석**:")
        st.markdown("- **✓ (흰색)**: 참고 가능한 위치 (j ≤ i)")
        st.markdown("- **✗ (파란색)**: 참고 불가능한 위치 (j > i)")
        st.markdown("- **대각선**: 자기 자신은 항상 참고 가능")
        st.markdown("- **아래쪽 삼각형**: 이전 토큰들은 참고 가능")
        st.markdown("- **위쪽 삼각형**: 미래 토큰들은 참고 불가능")
        
        # 마스킹 적용 예시
        st.markdown("### 📊 마스킹 적용 예시")
        
        # 예시 스코어 생성
        np.random.seed(42)
        example_scores = np.random.normal(0, 1, (seq_len, seq_len))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**마스킹 전 스코어**")
            st.dataframe(np_to_df(example_scores, row_idx=tgt_tokens, col_idx=tgt_tokens))
        
        with col2:
            st.markdown("**마스킹 후 스코어**")
            masked_scores = np.where(mask, -1e9, example_scores)
            st.dataframe(np_to_df(masked_scores, row_idx=tgt_tokens, col_idx=tgt_tokens))
            st.markdown("**참고**: `-1e9`는 softmax 후 0이 되는 값")
        
        # Softmax 적용 결과
        st.markdown("### 🎲 Softmax 적용 결과")
        
        # 마스킹 전후 softmax 비교
        softmax_before = softmax(example_scores, axis=-1)
        softmax_after = softmax(masked_scores, axis=-1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**마스킹 전 Softmax**")
            st.dataframe(np_to_df(softmax_before, row_idx=tgt_tokens, col_idx=tgt_tokens))
            st.markdown("**문제**: 미래 토큰에 대한 어텐션이 0이 아님")
        
        with col2:
            st.markdown("**마스킹 후 Softmax**")
            st.dataframe(np_to_df(softmax_after, row_idx=tgt_tokens, col_idx=tgt_tokens))
            st.markdown("**해결**: 미래 토큰에 대한 어텐션이 0이 됨")
        
        # 인과성의 중요성
        st.markdown("### 🎯 인과성의 중요성")
        st.markdown("""
        **학습 시**: 
        - 정답을 미리 보면 안 됨 (Cheating 방지)
        - 실제 추론 상황과 동일한 조건 필요
        
        **추론 시**: 
        - 한 번에 하나씩 토큰 생성
        - 이전에 생성된 토큰들만 참고 가능
        
        **예시**: "I ate" 다음에 "a"를 생성할 때
        - ✅ "I", "ate"는 참고 가능
        - ❌ "meal"은 참고 불가능 (아직 생성되지 않음)
        """)

# --------------------------------------------------
# 탭 5: 어텐션 지도
# --------------------------------------------------
with tabs[4]:
    st.subheader("어텐션 지도(Heatmap) 상세 분석")
    if analyze:
        # 크로스 어텐션 계산
        Qd = linear_projection(tgt_E, Wq)
        Ks = linear_projection(src_E, Wk)
        Vs = linear_projection(src_E, Wv)
        _, Wd, Sd = scaled_dot_product_attention(Qd, Ks, Vs)
        
        # 1. 어텐션 매트릭스 히트맵
        st.markdown("### 1. 어텐션 매트릭스")
        plot_heatmap(Wd, xticks=src_tokens, yticks=tgt_tokens, title="Cross-Attention Matrix")
        st.markdown("**읽는 법**: 행=타깃(영어) 토큰, 열=소스(한국어) 토큰. 값이 클수록 해당 타깃 단어가 그 소스 단어를 더 참고합니다.")
        
        # 2. 원본단어별 총 어텐션 (열 합계)
        st.markdown("### 2. 원본단어별 총 어텐션")
        src_total_attention = np.sum(Wd, axis=0)  # 각 소스 토큰이 받는 총 어텐션
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(src_tokens, src_total_attention)
        ax.set_title("각 원본 단어가 받는 총 어텐션", fontsize=12)
        ax.set_ylabel("총 어텐션 가중치", fontsize=10)
        ax.set_xlabel("원본 단어", fontsize=10)
        # 값 표시
        for bar, value in zip(bars, src_total_attention):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{value:.3f}', ha='center', va='bottom')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # 3. 번역 단어별 어텐션 분산
        st.markdown("### 3. 번역 단어별 어텐션 분산")
        tgt_attention_variance = np.var(Wd, axis=1)  # 각 타깃 토큰의 어텐션 분산
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(tgt_tokens, tgt_attention_variance)
        ax.set_title("각 번역 단어의 어텐션 분산", fontsize=12)
        ax.set_ylabel("분산", fontsize=10)
        ax.set_xlabel("번역 단어", fontsize=10)
        # 값 표시
        for bar, value in zip(bars, tgt_attention_variance):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                   f'{value:.3f}', ha='center', va='bottom')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # 4. 엔트로피 계산
        st.markdown("### 4. 어텐션 엔트로피")
        def calculate_entropy(attention_weights):
            """어텐션 가중치의 엔트로피 계산"""
            # 0에 가까운 값들을 작은 양수로 대체
            eps = 1e-10
            weights = np.maximum(attention_weights, eps)
            # 정규화
            weights = weights / np.sum(weights, axis=-1, keepdims=True)
            # 엔트로피 계산: -sum(p * log(p))
            entropy = -np.sum(weights * np.log(weights), axis=-1)
            return entropy
        
        # 각 타깃 토큰별 엔트로피
        tgt_entropy = calculate_entropy(Wd)
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(tgt_tokens, tgt_entropy)
        ax.set_title("각 번역 단어의 어텐션 엔트로피", fontsize=12)
        ax.set_ylabel("엔트로피", fontsize=10)
        ax.set_xlabel("번역 단어", fontsize=10)
        # 값 표시
        for bar, value in zip(bars, tgt_entropy):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{value:.3f}', ha='center', va='bottom')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # 5. 통계 요약
        st.markdown("### 5. 어텐션 통계 요약")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("평균 어텐션", f"{np.mean(Wd):.4f}")
            st.metric("최대 어텐션", f"{np.max(Wd):.4f}")
            st.metric("최소 어텐션", f"{np.min(Wd):.4f}")
        
        with col2:
            st.metric("어텐션 표준편차", f"{np.std(Wd):.4f}")
            st.metric("가장 집중된 타깃", tgt_tokens[np.argmin(tgt_entropy)])
            st.metric("가장 분산된 타깃", tgt_tokens[np.argmax(tgt_entropy)])
        
        with col3:
            st.metric("가장 주목받는 소스", src_tokens[np.argmax(src_total_attention)])
            st.metric("가장 주목받는 타깃", tgt_tokens[np.argmax(np.sum(Wd, axis=1))])
            st.metric("전체 엔트로피", f"{np.mean(tgt_entropy):.4f}")
        
        # 6. 어텐션 패턴 분석
        st.markdown("### 6. 어텐션 패턴 분석")
        st.markdown("**높은 엔트로피 (분산된 어텐션)**: 해당 타깃 단어가 여러 소스 단어에 고르게 주목")
        st.markdown("**낮은 엔트로피 (집중된 어텐션)**: 해당 타깃 단어가 특정 소스 단어에 집중")
        st.markdown("**높은 분산**: 해당 타깃 단어의 어텐션이 불균등하게 분포")
        st.markdown("**낮은 분산**: 해당 타깃 단어의 어텐션이 균등하게 분포")

# --------------------------------------------------
# 탭 6: 임베딩 분석
# --------------------------------------------------
with tabs[5]:
    st.subheader("임베딩 유사도 & 분포")
    if analyze:
        all_tokens = ["[SRC]:"+t for t in src_tokens] + ["[TGT]:"+t for t in tgt_tokens]
        X = np.vstack([src_E, tgt_E])
        # 코사인 유사도
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        cos = Xn @ Xn.T
        plot_heatmap(cos, xticks=all_tokens, yticks=all_tokens, title="코사인 유사도")

        st.markdown("### PCA 2D 분포(시연용)")
        Z = (X - X.mean(0, keepdims=True))
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        Z2 = Z @ Vt.T[:, :2]
        fig, ax = plt.subplots()
        ax.scatter(Z2[:len(src_tokens), 0], Z2[:len(src_tokens), 1], label="SRC")
        ax.scatter(Z2[len(src_tokens):, 0], Z2[len(src_tokens):, 1], label="TGT")
        for i, t in enumerate(all_tokens):
            ax.annotate(t, (Z2[i,0], Z2[i,1]), fontsize=8)
        ax.set_title("임베딩 분포 (PCA 2D)", fontsize=12)
        ax.set_xlabel("첫 번째 주성분", fontsize=10)
        ax.set_ylabel("두 번째 주성분", fontsize=10)
        ax.legend()
        st.pyplot(fig)

# --------------------------------------------------
# 탭 7: PyTorch 구현
# --------------------------------------------------
with tabs[6]:
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
# 탭 8: AI 챗봇 (간단 예시)
# --------------------------------------------------
with tabs[7]:
    st.subheader("🔬 분석 결과에 대해 질문해보세요!")

    if analyze:
        # 어텐션 계산 (기존 로직 재사용)
        Qd = linear_projection(tgt_E, Wq)
        Ks = linear_projection(src_E, Wk)
        Vs = linear_projection(src_E, Wv)
        _, Wd, _ = scaled_dot_product_attention(Qd, Ks, Vs)
        
        st.markdown("### 📊 분석 결과 기반 질문")
        
        if len(tgt_tokens) > 0 and len(src_tokens) > 0:
            q1 = f"'{tgt_tokens[0]}' 단어가 가장 주목한 원본 단어는 무엇인가요?"
            if st.button(q1):
                focused_idx = np.argmax(Wd[0])
                focused_word = src_tokens[focused_idx]
                st.write(f"**답변:** '{focused_word}' 단어입니다. (어텐션 스코어: {Wd[0, focused_idx]:.3f})")

            if len(src_tokens) > 1:
                q2 = f"'{src_tokens[1]}' 단어는 어떤 번역 단어로부터 가장 많은 주목을 받았나요?"
                if st.button(q2):
                    attending_idx = np.argmax(Wd[:, 1])
                    attending_word = tgt_tokens[attending_idx]
                    st.write(f"**답변:** '{attending_word}' 단어입니다. (어텐션 스코어: {Wd[attending_idx, 1]:.3f})")
            
            # 추가 질문들
            q3 = "가장 높은 어텐션 스코어를 가진 단어 쌍은 무엇인가요?"
            if st.button(q3):
                max_idx = np.unravel_index(np.argmax(Wd), Wd.shape)
                max_score = Wd[max_idx]
                tgt_word = tgt_tokens[max_idx[0]]
                src_word = src_tokens[max_idx[1]]
                st.write(f"**답변:** '{tgt_word}' → '{src_word}' (어텐션 스코어: {max_score:.3f})")
            
            q4 = "어떤 영어 단어가 가장 분산된 어텐션을 보이나요?"
            if st.button(q4):
                # 엔트로피 계산
                def calculate_entropy(attention_weights):
                    eps = 1e-10
                    weights = np.maximum(attention_weights, eps)
                    weights = weights / np.sum(weights, axis=-1, keepdims=True)
                    entropy = -np.sum(weights * np.log(weights), axis=-1)
                    return entropy
                
                entropies = calculate_entropy(Wd)
                most_distributed_idx = np.argmax(entropies)
                most_distributed_word = tgt_tokens[most_distributed_idx]
                st.write(f"**답변:** '{most_distributed_word}' 단어입니다. (엔트로피: {entropies[most_distributed_idx]:.3f})")
        
        st.markdown("---")
        st.markdown("### 💬 자유 질문")
        
        # 사용자가 직접 질문할 수 있는 기능도 유지
        user_input = st.text_input("자유롭게 질문해보세요 (예: 어텐션이란?):", "")
        if st.button("질문하기"):
            # 간단한 키워드 기반 응답 (실제로는 더 복잡한 모델 사용)
            responses = {
                "어텐션": "어텐션 메커니즘은 입력 시퀀스의 특정 부분에 집중하여 출력을 생성하는 방법입니다.",
                "transformer": "Transformer는 어텐션 메커니즘을 기반으로 한 신경망 아키텍처입니다.",
                "self-attention": "Self-attention은 같은 시퀀스 내의 다른 위치들을 참조하는 어텐션입니다.",
                "cross-attention": "Cross-attention은 서로 다른 시퀀스 간의 어텐션입니다.",
                "멀티헤드": "멀티헤드 어텐션은 여러 개의 어텐션을 병렬로 수행하여 다양한 관계를 동시에 파악하는 방법입니다.",
                "마스킹": "마스킹은 디코더에서 미래 토큰을 보지 못하게 하여 학습 시 정답을 미리 엿보지 못하게 하는 기법입니다.",
                "q": "Query(질문)는 현재 단어가 다른 단어들과 어떤 관계를 맺어야 할지 알아보기 위해 던지는 질문입니다.",
                "k": "Key(키)는 다른 단어들이 '나 이런 정보를 가지고 있어!'라고 알려주는 역할을 합니다.",
                "v": "Value(값)는 실제 내용으로, Query와 가장 잘 맞는 Key에 연결된 정보를 가져옵니다."
            }
            
            response = "죄송합니다. 질문에 대한 답변을 찾을 수 없습니다. '어텐션', 'transformer', 'self-attention', 'cross-attention', '멀티헤드', '마스킹', 'q', 'k', 'v' 등에 대해 질문해보세요."
            for keyword, resp in responses.items():
                if keyword.lower() in user_input.lower():
                    response = resp
                    break
            
            st.write("**AI 응답:**", response)
            st.info("💡 이는 교육용 데모입니다. 실제 AI 챗봇은 더 정교한 어텐션 메커니즘을 사용합니다.")

    else:
        st.warning("먼저 '분석 시작' 버튼을 눌러주세요.")

# --------------------------------------------------
# 탭 9: 퀴즈
# --------------------------------------------------
with tabs[8]:
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
# 탭 10: 용어사전
# --------------------------------------------------
with tabs[9]:
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