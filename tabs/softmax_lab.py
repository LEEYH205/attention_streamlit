import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.common import linear_projection, softmax

def render_softmax_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv):
    """소프트맥스 실험실 탭을 렌더링합니다."""
    
    st.subheader("🧪 소프트맥스 실험실")
    st.markdown("""
    소프트맥스의 온도(τ)와 top-k 필터링을 실험해보세요. 
    어텐션 가중치가 어떻게 변화하는지 직접 확인할 수 있습니다.
    """)
    
    if analyze:
        # 어텐션 계산을 위한 데이터 준비
        Qd = linear_projection(tgt_E, Wq)
        Ks = linear_projection(src_E, Wk)
        Vs = linear_projection(src_E, Wv)
        
        # 기본 어텐션 스코어 계산
        dk = Qd.shape[-1]
        scores = Qd @ Ks.T / np.sqrt(dk)
        
        st.markdown("### 🎛️ 실험 파라미터 조정")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 온도(τ) 슬라이더
            # 동적 키 생성으로 중복 방지
            if 'softmax_counter' not in st.session_state:
                st.session_state.softmax_counter = 0
            st.session_state.softmax_counter += 1
            
            tau = st.slider(
                "Softmax Temperature (τ)", 
                min_value=0.1, 
                max_value=3.0, 
                value=1.0, 
                step=0.1,
                help="τ가 작을수록 어텐션이 더 집중되고, 클수록 더 분산됩니다.",
                key=f"softmax_tau_slider_{st.session_state.softmax_counter}"
            )
            
            st.markdown(f"**현재 온도: {tau}**")
            if tau < 1.0:
                st.info("🔍 **낮은 온도**: 어텐션이 특정 토큰에 집중")
            elif tau > 1.0:
                st.info("🌊 **높은 온도**: 어텐션이 여러 토큰에 분산")
            else:
                st.info("⚖️ **기본 온도**: 표준 소프트맥스")
        
        with col2:
            # top-k 슬라이더
            max_k = min(len(src_tokens), 10)
            topk = st.slider(
                "Top-k 필터링", 
                min_value=0, 
                max_value=max_k, 
                value=0, 
                step=1,
                help="0=끄기, k=가장 높은 k개 가중치만 남기기",
                key=f"softmax_topk_slider_{st.session_state.softmax_counter}"
            )
            
            st.markdown(f"**현재 설정: {topk if topk > 0 else '끄기'}**")
            if topk > 0:
                st.info(f"🎯 **Top-{topk}**: 가장 높은 {topk}개 가중치만 유지")
            else:
                st.info("📊 **전체**: 모든 가중치 표시")
        
        st.markdown("---")
        
        # 어텐션 가중치 계산
        st.markdown("### 📊 어텐션 가중치 계산")
        
        # 온도 적용
        scores_scaled = scores / tau
        
        # 소프트맥스 적용
        attention_weights = softmax(scores_scaled, axis=-1)
        
        # Top-k 필터링 적용
        if topk > 0:
            # 각 행에서 top-k만 남기고 나머지는 0으로
            filtered_weights = np.zeros_like(attention_weights)
            for i in range(len(tgt_tokens)):
                top_indices = np.argsort(attention_weights[i])[-topk:]
                filtered_weights[i, top_indices] = attention_weights[i, top_indices]
            
            # 정규화 (합이 1이 되도록)
            row_sums = filtered_weights.sum(axis=1, keepdims=True)
            filtered_weights = filtered_weights / (row_sums + 1e-9)
            final_weights = filtered_weights
        else:
            final_weights = attention_weights
        
        # 결과 시각화
        st.markdown("### 🎨 결과 시각화")
        
        # 히트맵
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(final_weights, cmap='viridis')
        
        # 축 라벨
        ax.set_xticks(np.arange(len(src_tokens)))
        ax.set_yticks(np.arange(len(tgt_tokens)))
        ax.set_xticklabels(src_tokens, rotation=45, ha="right")
        ax.set_yticklabels(tgt_tokens)
        
        # 제목
        title = f"어텐션 가중치 (τ={tau}"
        if topk > 0:
            title += f", Top-{topk}"
        title += ")"
        ax.set_title(title)
        
        # 값 주석
        for i in range(len(tgt_tokens)):
            for j in range(len(src_tokens)):
                value = final_weights[i, j]
                if value > 0.01:  # 의미있는 값만 표시
                    ax.text(j, i, f'{value:.3f}', ha="center", va="center",
                           color='white' if value > 0.3 else 'black', fontsize=9)
        
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)
        plt.close(fig)  # 메모리 절약을 위해 그래프 자동 정리
        
        # 통계 정보
        st.markdown("### 📈 통계 정보")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("평균 어텐션", f"{np.mean(final_weights):.4f}")
        with col2:
            st.metric("최대 어텐션", f"{np.max(final_weights):.4f}")
        with col3:
            st.metric("어텐션 엔트로피", f"{calculate_entropy(final_weights):.4f}")
        
        # 상세 분석
        st.markdown("### 🔍 상세 분석")
        
        # 각 타겟 토큰별로 가장 집중하는 소스 토큰 표시
        st.markdown("**각 영어 단어가 한국어 단어에 집중하는 정도:**")
        
        for i, tgt_token in enumerate(tgt_tokens):
            weights = final_weights[i]
            top_indices = np.argsort(weights)[::-1]  # 내림차순
            
            st.markdown(f"**{tgt_token}**:")
            for j, idx in enumerate(top_indices[:3]):  # 상위 3개
                weight = weights[idx]
                if weight > 0.01:  # 의미있는 값만
                    st.markdown(f"  - {src_tokens[idx]}: {weight:.3f}")
        
        # 온도 효과 설명
        st.markdown("### 🌡️ 온도(τ) 효과 설명")
        st.markdown("""
        **온도는 소프트맥스의 '집중도'를 조절합니다:**
        
        - **τ < 1.0 (낮은 온도)**: 
          - 어텐션이 특정 토큰에 매우 집중
          - 확률 분포가 뾰족해짐 (극단적)
          - 예: [0.9, 0.05, 0.03, 0.02]
        
        - **τ = 1.0 (기본 온도)**:
          - 표준 소프트맥스
          - 균형잡힌 어텐션 분포
        
        - **τ > 1.0 (높은 온도)**:
          - 어텐션이 여러 토큰에 분산
          - 확률 분포가 평평해짐 (균등)
          - 예: [0.3, 0.25, 0.25, 0.2]
        """)
        
        # Top-k 효과 설명
        if topk > 0:
            st.markdown("### 🎯 Top-k 필터링 효과 설명")
            st.markdown(f"""
            **Top-{topk} 필터링은 가장 중요한 연결만 남깁니다:**
            
            - **장점**: 
              - 노이즈 제거
              - 해석 가능성 향상
              - 계산 효율성 증가
            
            - **주의사항**:
              - 정보 손실 가능성
              - 너무 작은 k는 성능 저하
              - 적절한 k 선택이 중요
            """)
    
    else:
        st.warning("먼저 '분석 시작' 버튼을 눌러주세요.")

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
