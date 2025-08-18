import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.common import linear_projection, softmax

def render_attention_map(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv):
    """어텐션 지도 탭을 렌더링합니다."""
    
    st.subheader("🗺️ 어텐션 지도")
    st.markdown("""
    **어텐션 지도(Attention Map)**는 모델이 각 단계에서 어떤 입력 토큰에 집중하고 있는지를 시각적으로 보여줍니다.
    이를 통해 모델의 "사고 과정"을 이해할 수 있습니다.
    """)
    
    if analyze:
        st.markdown("---")
        
        # 어텐션 계산
        Q = linear_projection(tgt_E, Wq)
        K = linear_projection(src_E, Wk)
        V = linear_projection(src_E, Wv)
        
        # 어텐션 스코어 및 가중치 계산
        dk = Q.shape[-1]
        scores = Q @ K.T / np.sqrt(dk)
        attention_weights = softmax(scores, axis=-1)
        
        # 시각화 옵션
        st.markdown("### 🎨 시각화 옵션")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 컬러맵 선택
            colormap = st.selectbox(
                "컬러맵 선택",
                ["viridis", "plasma", "inferno", "magma", "coolwarm", "RdBu_r"],
                help="어텐션 가중치를 표현할 컬러맵을 선택하세요"
            )
        
        with col2:
            # 값 표시 여부
            show_values = st.checkbox("값 표시", value=True, help="어텐션 가중치 값을 셀에 표시합니다")
        
        # 1. 기본 어텐션 지도
        st.markdown("### 🗺️ 기본 어텐션 지도")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(attention_weights, cmap=colormap, aspect='auto')
        
        # 축 라벨
        ax.set_xticks(np.arange(len(src_tokens)))
        ax.set_yticks(np.arange(len(tgt_tokens)))
        ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(tgt_tokens, fontsize=10)
        
        # 제목 및 라벨
        ax.set_title("어텐션 가중치 지도", fontsize=14, fontweight='bold')
        ax.set_xlabel("Source Tokens (한국어)", fontsize=12)
        ax.set_ylabel("Target Tokens (영어)", fontsize=12)
        
        # 값 주석 (옵션)
        if show_values:
            for i in range(len(tgt_tokens)):
                for j in range(len(src_tokens)):
                    value = attention_weights[i, j]
                    if value > 0.01:  # 의미있는 값만 표시
                        # 배경에 따른 텍스트 색상 결정
                        if colormap in ["viridis", "plasma", "inferno", "magma"]:
                            text_color = 'white' if value > 0.3 else 'black'
                        else:
                            text_color = 'black'
                        
                        ax.text(j, i, f'{value:.3f}', ha="center", va="center", 
                               color=text_color, fontsize=8, fontweight='bold')
        
        # 컬러바
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("어텐션 가중치", fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 2. 어텐션 패턴 분석
        st.markdown("### 🔍 어텐션 패턴 분석")
        
        # 각 타겟 토큰별로 가장 집중하는 소스 토큰
        st.markdown("**각 영어 단어가 가장 집중하는 한국어 단어:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            for i, tgt_token in enumerate(tgt_tokens):
                weights = attention_weights[i]
                top_indices = np.argsort(weights)[::-1]  # 내림차순
                
                st.markdown(f"**{tgt_token}**:")
                for j, idx in enumerate(top_indices[:3]):  # 상위 3개
                    weight = weights[idx]
                    if weight > 0.01:
                        st.markdown(f"  - {src_tokens[idx]}: {weight:.3f}")
        
        with col2:
            # 어텐션 집중도 통계
            st.markdown("**어텐션 집중도 통계:**")
            
            # 각 타겟 토큰의 최대 어텐션 값
            max_attentions = np.max(attention_weights, axis=1)
            for i, (token, max_att) in enumerate(zip(tgt_tokens, max_attentions)):
                st.metric(f"{token} 최대 어텐션", f"{max_att:.3f}")
        
        # 3. 어텐션 분포 히스토그램
        st.markdown("### 📊 어텐션 분포 히스토그램")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 전체 어텐션 가중치 분포
        ax1.hist(attention_weights.flatten(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title("전체 어텐션 가중치 분포", fontsize=12)
        ax1.set_xlabel("어텐션 가중치")
        ax1.set_ylabel("빈도")
        ax1.grid(True, alpha=0.3)
        
        # 각 타겟 토큰별 어텐션 분포
        for i, token in enumerate(tgt_tokens):
            ax2.hist(attention_weights[i], bins=20, alpha=0.6, label=token)
        
        ax2.set_title("타겟 토큰별 어텐션 분포", fontsize=12)
        ax2.set_xlabel("어텐션 가중치")
        ax2.set_ylabel("빈도")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 4. 어텐션 히트맵 상세 분석
        st.markdown("### 🔬 어텐션 히트맵 상세 분석")
        
        # 어텐션 강도별 분류
        st.markdown("**어텐션 강도별 분류:**")
        
        # 강한 어텐션 (> 0.5)
        strong_attention = attention_weights > 0.5
        strong_count = np.sum(strong_attention)
        
        # 중간 어텐션 (0.1 ~ 0.5)
        medium_attention = (attention_weights > 0.1) & (attention_weights <= 0.5)
        medium_count = np.sum(medium_attention)
        
        # 약한 어텐션 (≤ 0.1)
        weak_attention = attention_weights <= 0.1
        weak_count = np.sum(weak_attention)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("강한 어텐션 (>0.5)", f"{strong_count}개")
            if strong_count > 0:
                strong_positions = np.where(strong_attention)
                st.markdown("**위치:**")
                for i, j in zip(strong_positions[0], strong_positions[1]):
                    st.markdown(f"- {tgt_tokens[i]} → {src_tokens[j]}: {attention_weights[i, j]:.3f}")
        
        with col2:
            st.metric("중간 어텐션 (0.1~0.5)", f"{medium_count}개")
        
        with col3:
            st.metric("약한 어텐션 (≤0.1)", f"{weak_count}개")
        
        # 5. 어텐션 패턴 해석 가이드
        st.markdown("### 📖 어텐션 패턴 해석 가이드")
        
        st.markdown("""
        **어텐션 지도를 해석하는 방법:**
        
        - **밝은 색상**: 높은 어텐션 가중치 (강한 연결)
        - **어두운 색상**: 낮은 어텐션 가중치 (약한 연결)
        - **대각선 패턴**: 자기 자신에 대한 어텐션 (Self-Attention)
        - **수평선 패턴**: 특정 소스 토큰에 집중
        - **수직선 패턴**: 특정 타겟 토큰이 여러 소스에 분산
        
        **일반적인 패턴:**
        - **번역**: 의미적으로 대응하는 단어들 간의 강한 연결
        - **문법**: 문법적 관계가 있는 단어들 간의 연결
        - **문맥**: 전체적인 문맥을 이해하기 위한 분산된 연결
        """)
        
        # 6. 인터랙티브 분석
        st.markdown("### 🎯 인터랙티브 분석")
        
        # 특정 타겟 토큰 선택
        selected_target = st.selectbox(
            "분석할 타겟 토큰 선택",
            tgt_tokens,
            help="특정 영어 단어의 어텐션 패턴을 자세히 분석합니다"
        )
        
        if selected_target:
            target_idx = tgt_tokens.index(selected_target)
            target_weights = attention_weights[target_idx]
            
            # 선택된 타겟의 어텐션 패턴
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(src_tokens, target_weights, color='lightcoral', alpha=0.7)
            ax.set_title(f"'{selected_target}'의 어텐션 패턴", fontsize=14, fontweight='bold')
            ax.set_xlabel("Source Tokens (한국어)")
            ax.set_ylabel("어텐션 가중치")
            ax.set_ylim(0, 1)
            
            # 값 주석
            for bar, weight in zip(bars, target_weights):
                if weight > 0.01:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{weight:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # 상세 분석
            st.markdown(f"**'{selected_target}' 어텐션 분석:**")
            
            # 상위 3개 소스 토큰
            top_sources = np.argsort(target_weights)[::-1][:3]
            st.markdown("**가장 집중하는 소스 토큰:**")
            for i, idx in enumerate(top_sources):
                weight = target_weights[idx]
                if weight > 0.01:
                    st.markdown(f"{i+1}. **{src_tokens[idx]}**: {weight:.3f}")
    
    else:
        st.warning("먼저 '분석 시작' 버튼을 눌러주세요.")
