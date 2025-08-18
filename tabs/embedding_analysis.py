import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# hash_vec은 app.py에 정의되어 있음

def render_embedding_analysis(analyze, src_tokens, tgt_tokens, src_E, tgt_E):
    """임베딩 분석 탭을 렌더링합니다."""
    
    st.subheader("🔍 임베딩 분석")
    st.markdown("""
    **임베딩(Embedding)**은 단어나 토큰을 고차원 벡터 공간으로 변환한 것입니다.
    이 공간에서 의미적으로 유사한 단어들은 가까이 위치하게 됩니다.
    """)
    
    if analyze:
        st.markdown("---")
        
        # 시각화 방법 선택
        visualization_method = st.selectbox(
            "시각화 방법 선택",
            ["PCA (Principal Component Analysis)", "t-SNE (t-Distributed Stochastic Neighbor Embedding)"],
            help="고차원 임베딩을 2D로 시각화하는 방법을 선택하세요"
        )
        
        # 모든 토큰과 임베딩 결합
        all_tokens = src_tokens + tgt_tokens
        all_embeddings = np.vstack([src_E, tgt_E])
        
        st.markdown(f"### 📊 {visualization_method}")
        st.markdown(f"**분석 대상**: {len(all_tokens)}개 토큰, {all_embeddings.shape[1]}차원 임베딩")
        
        # 차원 축소
        if visualization_method == "PCA":
            # PCA 적용
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(all_embeddings)
            
            # 설명된 분산 비율
            explained_variance = pca.explained_variance_ratio_
            st.info(f"**PCA 설명된 분산**: PC1: {explained_variance[0]:.3f} ({explained_variance[0]*100:.1f}%), PC2: {explained_variance[1]:.3f} ({explained_variance[1]*100:.1f}%)")
            
        else:  # t-SNE
            # t-SNE 적용 (계산 시간이 오래 걸릴 수 있음)
            with st.spinner("t-SNE 계산 중... (잠시 기다려주세요)"):
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_tokens)-1))
                reduced_embeddings = tsne.fit_transform(all_embeddings)
            
            st.success("✅ t-SNE 계산 완료!")
        
        # 시각화
        st.markdown("### 🎨 임베딩 공간 시각화")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 소스 토큰 (한국어) - 파란색
        ax.scatter(reduced_embeddings[:len(src_tokens), 0], 
                  reduced_embeddings[:len(src_tokens), 1], 
                  c='blue', s=100, alpha=0.7, label='Source (한국어)')
        
        # 타겟 토큰 (영어) - 빨간색
        ax.scatter(reduced_embeddings[len(src_tokens):, 0], 
                  reduced_embeddings[len(src_tokens):, 1], 
                  c='red', s=100, alpha=0.7, label='Target (영어)')
        
        # 토큰 라벨 추가
        for i, token in enumerate(all_tokens):
            ax.annotate(token, 
                       (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold')
        
        # 축 라벨
        if visualization_method == "PCA":
            ax.set_xlabel(f"Principal Component 1 ({explained_variance[0]*100:.1f}%)")
            ax.set_ylabel(f"Principal Component 2 ({explained_variance[1]*100:.1f}%)")
        else:
            ax.set_xlabel("t-SNE Dimension 1")
            ax.set_ylabel("t-SNE Dimension 2")
        
        ax.set_title("토큰 임베딩 공간", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 2. 임베딩 유사도 분석
        st.markdown("### 🔗 임베딩 유사도 분석")
        
        # 코사인 유사도 계산
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(all_embeddings)
        
        # 유사도 히트맵
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(similarities, cmap='RdYlBu_r', vmin=-1, vmax=1)
        
        # 축 라벨
        ax.set_xticks(np.arange(len(all_tokens)))
        ax.set_yticks(np.arange(len(all_tokens)))
        ax.set_xticklabels(all_tokens, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(all_tokens, fontsize=8)
        
        # 제목
        ax.set_title("토큰 간 코사인 유사도", fontsize=12)
        
        # 컬러바
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("코사인 유사도")
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 3. 가장 유사한 토큰 쌍 찾기
        st.markdown("### 🎯 가장 유사한 토큰 쌍")
        
        # 대각선 제외하고 가장 유사한 쌍 찾기
        np.fill_diagonal(similarities, -2)  # 대각선을 매우 낮은 값으로 설정
        
        # 상위 5개 유사한 쌍
        top_pairs = []
        for _ in range(5):
            max_idx = np.unravel_index(np.argmax(similarities), similarities.shape)
            max_sim = similarities[max_idx]
            if max_sim > 0.1:  # 의미있는 유사도만
                top_pairs.append((max_idx, max_sim))
                similarities[max_idx] = -2  # 이미 선택된 쌍 제외
        
        if top_pairs:
            st.markdown("**가장 유사한 토큰 쌍 (상위 5개):**")
            for i, ((idx1, idx2), sim) in enumerate(top_pairs):
                token1, token2 = all_tokens[idx1], all_tokens[idx2]
                lang1 = "한국어" if idx1 < len(src_tokens) else "영어"
                lang2 = "한국어" if idx2 < len(src_tokens) else "영어"
                st.markdown(f"{i+1}. **{token1}** ({lang1}) ↔ **{token2}** ({lang2}): {sim:.3f}")
        
        # 4. 임베딩 통계 분석
        st.markdown("### 📈 임베딩 통계 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 임베딩 차원별 통계
            st.markdown("**임베딩 차원별 통계:**")
            
            mean_embeddings = np.mean(all_embeddings, axis=0)
            std_embeddings = np.std(all_embeddings, axis=0)
            
            st.metric("평균 임베딩 크기", f"{np.mean(np.linalg.norm(all_embeddings, axis=1)):.3f}")
            st.metric("임베딩 표준편차", f"{np.mean(std_embeddings):.3f}")
            st.metric("최대 임베딩 크기", f"{np.max(np.linalg.norm(all_embeddings, axis=1)):.3f}")
            st.metric("최소 임베딩 크기", f"{np.min(np.linalg.norm(all_embeddings, axis=1)):.3f}")
        
        with col2:
            # 언어별 통계
            st.markdown("**언어별 임베딩 통계:**")
            
            src_norms = np.linalg.norm(src_E, axis=1)
            tgt_norms = np.linalg.norm(tgt_E, axis=1)
            
            st.metric("한국어 평균 크기", f"{np.mean(src_norms):.3f}")
            st.metric("영어 평균 크기", f"{np.mean(tgt_norms):.3f}")
            st.metric("한국어 표준편차", f"{np.std(src_norms):.3f}")
            st.metric("영어 표준편차", f"{np.std(tgt_norms):.3f}")
        
        # 5. 임베딩 분포 히스토그램
        st.markdown("### 📊 임베딩 분포 히스토그램")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 임베딩 크기 분포
        embedding_norms = np.linalg.norm(all_embeddings, axis=1)
        ax1.hist(embedding_norms, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax1.set_title("임베딩 벡터 크기 분포", fontsize=12)
        ax1.set_xlabel("임베딩 크기 (L2 norm)")
        ax1.set_ylabel("빈도")
        ax1.grid(True, alpha=0.3)
        
        # 언어별 임베딩 크기 비교
        ax2.hist(src_norms, bins=15, alpha=0.6, label='한국어', color='blue')
        ax2.hist(tgt_norms, bins=15, alpha=0.6, label='영어', color='red')
        ax2.set_title("언어별 임베딩 크기 비교", fontsize=12)
        ax2.set_xlabel("임베딩 크기 (L2 norm)")
        ax2.set_ylabel("빈도")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 6. 임베딩 품질 평가
        st.markdown("### 🎯 임베딩 품질 평가")
        
        # 임베딩 품질 지표
        st.markdown("**임베딩 품질 지표:**")
        
        # 1. 임베딩 크기 일관성
        norm_std = np.std(embedding_norms)
        norm_cv = norm_std / np.mean(embedding_norms)  # 변동계수
        
        if norm_cv < 0.3:
            norm_quality = "🟢 우수"
        elif norm_cv < 0.5:
            norm_quality = "🟡 보통"
        else:
            norm_quality = "🔴 개선 필요"
        
        st.metric("임베딩 크기 일관성", norm_quality, f"변동계수: {norm_cv:.3f}")
        
        # 2. 언어 간 유사도
        cross_language_sim = cosine_similarity(src_E, tgt_E)
        avg_cross_sim = np.mean(cross_language_sim)
        
        if avg_cross_sim > 0.3:
            cross_quality = "🟢 우수"
        elif avg_cross_sim > 0.1:
            cross_quality = "🟡 보통"
        else:
            cross_quality = "🔴 개선 필요"
        
        st.metric("언어 간 유사도", cross_quality, f"평균: {avg_cross_sim:.3f}")
        
        # 3. 임베딩 공간 활용도
        # PCA로 설명된 분산 비율
        pca_full = PCA()
        pca_full.fit(all_embeddings)
        explained_variance_ratio = pca_full.explained_variance_ratio_
        
        # 상위 10개 성분으로 설명되는 분산 비율
        top_10_variance = np.sum(explained_variance_ratio[:10])
        
        if top_10_variance > 0.8:
            space_quality = "🟢 우수"
        elif top_10_variance > 0.6:
            space_quality = "🟡 보통"
        else:
            space_quality = "🔴 개선 필요"
        
        st.metric("공간 활용도", space_quality, f"상위 10개 성분: {top_10_variance:.1%}")
        
        # 7. 임베딩 개선 제안
        st.markdown("### 💡 임베딩 개선 제안")
        
        st.markdown("""
        **현재 임베딩의 특징과 개선 방향:**
        
        - **임베딩 크기**: 모든 토큰이 비슷한 크기를 가지면 안정적
        - **언어 간 유사도**: 번역 쌍이 의미적으로 유사해야 함
        - **공간 활용도**: 고차원 공간을 효율적으로 활용해야 함
        
        **개선 방법:**
        - **정규화**: 임베딩 벡터를 정규화하여 크기 일관성 확보
        - **대조 학습**: 번역 쌍 간의 유사도를 높이는 학습
        - **차원 축소**: 불필요한 차원 제거로 노이즈 감소
        """)
    
    else:
        st.warning("먼저 '분석 시작' 버튼을 눌러주세요.")
