import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.common import linear_projection, softmax

def render_masking_causality(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv):
    """마스킹 & 인과성 탭을 렌더링합니다."""
    
    st.subheader("⛔ 마스킹 & 인과성")
    st.markdown("""
    **마스킹(Masking)**은 어텐션이 특정 위치의 토큰을 "보지 못하게" 하는 기법입니다.
    주로 **디코더**에서 사용되어, 현재 위치보다 미래의 토큰에 어텐션하지 못하게 합니다.
    
    **인과성(Causality)**은 모델이 미래 정보를 사용하지 못하게 하여, 실제 추론 시와 동일한 조건을 만드는 것입니다.
    """)
    
    if analyze:
        st.markdown("---")
        
        # 어텐션 타입 선택
        # 동적 키 생성으로 중복 방지
        if 'masking_counter' not in st.session_state:
            st.session_state.masking_counter = 0
        st.session_state.masking_counter += 1
        
        attention_type = st.radio(
            "어텐션 타입 선택",
            ["Self-Attention (마스킹 없음)", "Masked Self-Attention (인과성 보장)"],
            key=f"masking_attention_type_radio_{st.session_state.masking_counter}",
            help="마스킹의 효과를 비교해보세요"
        )
        
        st.markdown(f"### 🎯 {attention_type}")
        
        # Q, K, V 계산
        Q = linear_projection(tgt_E, Wq)
        K = linear_projection(tgt_E, Wk)
        V = linear_projection(tgt_E, Wv)
        
        # 어텐션 스코어 계산
        dk = Q.shape[-1]
        scores = Q @ K.T / np.sqrt(dk)
        
        # 마스킹 적용 여부
        if "Masked" in attention_type:
            # 하삼각 행렬 마스크 생성 (인과성 보장)
            mask = np.triu(np.ones_like(scores), k=1)
            scores = scores - 1e9 * mask  # 마스킹된 위치를 매우 작은 값으로
            st.info("🔒 **마스킹 적용**: 현재 위치보다 미래의 토큰은 어텐션하지 못합니다.")
        else:
            st.info("👁️ **마스킹 없음**: 모든 토큰에 자유롭게 어텐션할 수 있습니다.")
        
        # 소프트맥스 적용
        attention_weights = softmax(scores, axis=-1)
        
        # 시각화
        st.markdown("### 🎨 어텐션 가중치 시각화")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 원본 스코어
        im1 = ax1.imshow(scores, cmap='RdBu_r', vmin=-np.max(np.abs(scores)), vmax=np.max(np.abs(scores)))
        ax1.set_title("어텐션 스코어 (마스킹 전)", fontsize=12)
        ax1.set_xticks(np.arange(len(tgt_tokens)))
        ax1.set_yticks(np.arange(len(tgt_tokens)))
        ax1.set_xticklabels(tgt_tokens, rotation=45, ha="right")
        ax1.set_yticklabels(tgt_tokens)
        ax1.set_xlabel("Key (영어)")
        ax1.set_ylabel("Query (영어)")
        
        # 값 주석
        for i in range(len(tgt_tokens)):
            for j in range(len(tgt_tokens)):
                value = scores[i, j]
                if abs(value) > 0.1:  # 의미있는 값만
                    color = 'white' if abs(value) > 1.0 else 'black'
                    ax1.text(j, i, f'{value:.2f}', ha="center", va="center", 
                            color=color, fontsize=8)
        
        plt.colorbar(im1, ax=ax1)
        
        # 최종 어텐션 가중치
        im2 = ax2.imshow(attention_weights, cmap='viridis')
        ax2.set_title("어텐션 가중치 (마스킹 후)", fontsize=12)
        ax2.set_xticks(np.arange(len(tgt_tokens)))
        ax2.set_yticks(np.arange(len(tgt_tokens)))
        ax2.set_xticklabels(tgt_tokens, rotation=45, ha="right")
        ax2.set_yticklabels(tgt_tokens)
        ax2.set_xlabel("Key (영어)")
        ax2.set_ylabel("Query (영어)")
        
        # 값 주석
        for i in range(len(tgt_tokens)):
            for j in range(len(tgt_tokens)):
                value = attention_weights[i, j]
                if value > 0.01:  # 의미있는 값만
                    color = 'white' if value > 0.3 else 'black'
                    ax2.text(j, i, f'{value:.3f}', ha="center", va="center", 
                            color=color, fontsize=8)
        
        plt.colorbar(im2, ax=ax2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)  # 메모리 절약을 위해 그래프 자동 정리
        
        # 마스킹 효과 분석
        st.markdown("### 🔍 마스킹 효과 분석")
        
        if "Masked" in attention_type:
            # 마스킹된 위치 확인
            masked_positions = np.where(mask == 1)
            st.markdown(f"**마스킹된 위치 수**: {len(masked_positions[0])}")
            
            # 각 토큰별로 어텐션할 수 있는 토큰 수
            attention_counts = []
            for i in range(len(tgt_tokens)):
                count = np.sum(attention_weights[i] > 0.01)
                attention_counts.append(count)
                st.markdown(f"- **{tgt_tokens[i]}**: {count}개 토큰에 어텐션")
            
            # 마스킹 효과 시각화
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(tgt_tokens, attention_counts, color='skyblue', alpha=0.7)
            ax.set_title("각 토큰이 어텐션할 수 있는 토큰 수", fontsize=12)
            ax.set_xlabel("토큰")
            ax.set_ylabel("어텐션 가능한 토큰 수")
            ax.set_ylim(0, len(tgt_tokens))
            
            # 값 주석
            for bar, count in zip(bars, attention_counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{count}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        # 인과성 설명
        st.markdown("### ⏰ 인과성(Causality)의 중요성")
        st.markdown("""
        **인과성이 중요한 이유:**
        
        1. **실제 추론과 일치**: 학습 시와 추론 시의 조건을 동일하게 만듦
        2. **정보 누출 방지**: 미래 정보를 사용하여 과거를 예측하는 것을 방지
        3. **자기회귀(Auto-regressive) 생성**: 한 번에 하나씩 토큰을 생성할 때 필수
        
        **예시:**
        - "I love" 다음에 "you"를 예측할 때
        - "you"에 대한 정보는 사용할 수 없음
        - 오직 "I"와 "love"만을 보고 예측해야 함
        """)
        
        # 마스킹 패턴 비교
        st.markdown("### 🔒 마스킹 패턴 비교")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Self-Attention (마스킹 없음)**")
            st.markdown("""
            - 모든 토큰이 서로를 볼 수 있음
            - 양방향 정보 활용
            - 인코더에서 주로 사용
            - 문맥 이해에 유리
            """)
        
        with col2:
            st.markdown("**Masked Self-Attention**")
            st.markdown("""
            - 현재 위치보다 미래 토큰은 보지 못함
            - 단방향 정보만 활용
            - 디코더에서 주로 사용
            - 순차적 생성에 필수
            """)
        
        # 실제 사용 예시
        st.markdown("### 💡 실제 사용 예시")
        st.markdown("""
        **Transformer 모델에서의 사용:**
        
        - **인코더**: Self-Attention (마스킹 없음)
          - 입력 문장의 모든 단어를 동시에 처리
          - 양방향 문맥 이해
        
        - **디코더**: Masked Self-Attention
          - 이미 생성된 토큰만 참조
          - 미래 토큰은 마스킹
          - 순차적 텍스트 생성
        """)
    
    else:
        st.warning("먼저 '분석 시작' 버튼을 눌러주세요.")
