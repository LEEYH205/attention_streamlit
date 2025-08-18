import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.common import linear_projection, calculate_entropy

def render_multihead_visualization(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv, dim):
    """멀티헤드 시각화 탭을 렌더링합니다."""
    
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
        # 동적 키 생성으로 중복 방지
        if 'multihead_counter' not in st.session_state:
            st.session_state.multihead_counter = 0
        st.session_state.multihead_counter += 1
        
        num_heads = st.slider("헤드 수 선택", min_value=1, max_value=8, value=4, help="분석할 헤드의 수를 선택하세요", key=f"multihead_num_heads_slider_{st.session_state.multihead_counter}")
        
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
            # 간단한 어텐션 계산 (실제 scaled_dot_product_attention 함수 대신)
            scores = Qd_heads[i] @ Ks_heads[i].T / np.sqrt(head_dim)
            weights = softmax(scores, axis=-1)
            context = weights @ Vs_heads[i]
            
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
            plt.close(fig)  # 메모리 절약을 위해 그래프 자동 정리
            
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
                    ax.text(x, y, f"{avg_weights[y, x]:.2f}", ha="center", va="center", fontsize=8)
            
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)
            plt.close(fig)  # 메모리 절약을 위해 그래프 자동 정리
            
            # 헤드 간 유사도 분석
            st.markdown("### 📊 헤드 간 유사도 분석")
            
            # 코사인 유사도 계산
            similarities = np.zeros((num_heads, num_heads))
            for i in range(num_heads):
                for j in range(num_heads):
                    if i == j:
                        similarities[i, j] = 1.0
                    else:
                        # 어텐션 가중치를 평면화하여 유사도 계산
                        flat_i = head_weights[i].flatten()
                        flat_j = head_weights[j].flatten()
                        cos_sim = np.dot(flat_i, flat_j) / (np.linalg.norm(flat_i) * np.linalg.norm(flat_j))
                        similarities[i, j] = cos_sim
            
            # 유사도 히트맵
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(similarities, cmap='RdYlBu_r', vmin=-1, vmax=1)
            ax.set_xticks(np.arange(num_heads))
            ax.set_yticks(np.arange(num_heads))
            ax.set_xticklabels([f"Head {i+1}" for i in range(num_heads)])
            ax.set_yticklabels([f"Head {i+1}" for i in range(num_heads)])
            ax.set_title("헤드 간 유사도 (코사인 유사도)", fontsize=12)
            
            # 값 주석
            for i in range(num_heads):
                for j in range(num_heads):
                    ax.text(j, i, f"{similarities[i, j]:.2f}", ha="center", va="center", fontsize=10)
            
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)
            plt.close(fig)  # 메모리 절약을 위해 그래프 자동 정리
            
            # 유사도 해석
            st.markdown("**유사도 해석:**")
            st.markdown("- **높은 유사도 (0.7~1.0)**: 두 헤드가 비슷한 패턴을 학습")
            st.markdown("- **중간 유사도 (0.3~0.7)**: 두 헤드가 부분적으로 유사한 패턴")
            st.markdown("- **낮은 유사도 (-0.3~0.3)**: 두 헤드가 서로 다른 패턴을 학습")
            st.markdown("- **음의 유사도 (-1.0~-0.3)**: 두 헤드가 반대 패턴을 학습")
        
        # 4. 멀티헤드 효과 설명
        st.markdown("### 🎭 멀티헤드 어텐션의 효과")
        st.markdown("""
        **멀티헤드 어텐션은 다음과 같은 이점을 제공합니다:**
        
        - **다양성**: 각 헤드가 서로 다른 관계를 학습
        - **안정성**: 단일 헤드의 실패가 전체에 미치는 영향 최소화
        - **표현력**: 복잡한 관계를 더 잘 모델링
        - **병렬화**: 여러 헤드를 동시에 계산 가능
        
        **실제 예시:**
        - **Head 1**: 주어-동사 관계에 집중
        - **Head 2**: 형용사-명사 관계에 집중
        - **Head 3**: 전치사-명사 관계에 집중
        - **Head 4**: 전체적인 문맥 관계에 집중
        """)
    
    else:
        st.warning("먼저 '분석 시작' 버튼을 눌러주세요.")

def softmax(x, axis=-1):
    """소프트맥스 함수"""
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)
