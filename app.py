
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
    if row_idx is not None:
        df.index = row_idx
    if col_idx is not None:
        df.columns = col_idx
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

tabs = st.tabs(["단계별 어텐션", "멀티헤드 어텐션", "어텐션 지도", "임베딩 분석", "PyTorch 구현", "AI 챗봇"])

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
# 탭 1: 단계별 어텐션
# --------------------------------------------------
with tabs[0]:
    st.subheader("1) Scaled Dot-Product Attention 단계별 계산")
    
    # Q, K, V에 대한 직관적 설명 추가
    with st.expander("🤔 Q, K, V가 뭔가요? (클릭하여 펼치기)"):
        st.markdown("""
        어텐션을 도서관에서 원하는 정보를 찾는 과정에 비유해볼 수 있습니다.
        
        - **Query (Q)**: **나의 질문 혹은 검색어**입니다. 현재 단어가 문장의 다른 단어들과 어떤 관계를 맺어야 할지 알아보기 위해 던지는 질문입니다.
        - **Key (K)**: 도서관에 있는 **책들의 '핵심 키워드'나 '꼬리표'** 와 같습니다. 다른 모든 단어들이 "나 이런 정보를 가지고 있어!"라고 알려주는 역할을 합니다.
        - **Value (V)**: 책의 **실제 '내용'** 입니다. 내 질문(Query)과 가장 관련이 깊은 키워드(Key)를 찾았다면, 그 키워드가 붙어있는 책의 실제 내용(Value)을 가져와 참고합니다.
        
        **결론적으로 어텐션은, 나의 Query와 가장 잘 맞는 Key를 찾아서, 해당 Key에 연결된 Value를 가져오는 과정입니다.**
        """)
    
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
        # A. 인코더 Self-Attention (소스→소스)
        st.markdown("### A. 인코더 Self-Attention (소스→소스)")
        st.markdown("**목적**: 한국어 문장 내에서 각 단어가 다른 단어들과 어떻게 연결되는지 학습")
        
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

        # 구분선
        st.markdown("---")
        
        # B. 인코더–디코더 Cross-Attention (타깃→소스)
        st.markdown("### B. 인코더–디코더 Cross-Attention (타깃→소스)")
        st.markdown("**목적**: 영어 단어를 생성할 때 어떤 한국어 단어를 참고할지 결정")
        
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

        # 구분선
        st.markdown("---")
        
        # C. 디코더 마스크드 Self-Attention (타깃→타깃)
        st.markdown("### C. 디코더 마스크드 Self-Attention (타깃→타깃)")
        st.markdown("**목적**: 영어 문장을 생성할 때, 현재 단어가 이전에 생성된 단어들만 참고하도록 하여 정답을 미리 엿보지 못하게 함")

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

# --------------------------------------------------
# 탭 2: 멀티헤드 어텐션
# --------------------------------------------------
with tabs[1]:
    st.subheader("💡 Multi-Head Attention의 원리")
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
        st.markdown("### 멀티헤드 크로스 어텐션 예시 (Heads = 4)")
        
        num_heads = 4
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
        
        # 2. 병렬 어텐션 & 시각화
        st.write(f"**2. 병렬 어텐션**: 각 헤드가 독립적으로 어텐션을 계산합니다.")
        
        cols = st.columns(num_heads)
        head_outputs = []
        for i in range(num_heads):
            with cols[i]:
                context, weights, _ = scaled_dot_product_attention(Qd_heads[i], Ks_heads[i], Vs_heads[i])
                head_outputs.append(context)
                
                fig, ax = plt.subplots()
                im = ax.imshow(weights)
                ax.set_xticks(np.arange(len(src_tokens)))
                ax.set_yticks(np.arange(len(tgt_tokens)))
                ax.set_xticklabels(src_tokens, rotation=90, ha="right", fontsize=8)
                ax.set_yticklabels(tgt_tokens, fontsize=8)
                ax.set_title(f"Head {i+1}", fontsize=10)
                st.pyplot(fig)

        st.markdown("**해석**: 각 헤드(Head)가 서로 다른 어텐션 패턴을 보이는 것을 확인할 수 있습니다. 어떤 헤드는 단어 대 단어 관계에, 다른 헤드는 좀 더 분산된 관계에 집중합니다.")
        
        # 3. 결합 및 최종 변환
        st.write("**3. 결합 및 최종 변환**: 결과들을 합치고 최종 출력 벡터를 만듭니다.")
        concatenated = np.concatenate([h.transpose(1,0) for h in head_outputs], axis=-1).reshape(len(tgt_tokens), dim)
        # Wo는 시연용으로 생략
        st.write(" - Concatenated shape:", concatenated.shape)
        st.dataframe(np_to_df(concatenated, row_idx=tgt_tokens))

# --------------------------------------------------
# 탭 3: 어텐션 지도
# --------------------------------------------------
with tabs[2]:
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
# 탭 4: 임베딩 분석
# --------------------------------------------------
with tabs[3]:
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
# 탭 5: PyTorch 구현
# --------------------------------------------------
with tabs[4]:
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
# 탭 6: AI 챗봇 (간단 예시)
# --------------------------------------------------
with tabs[5]:
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