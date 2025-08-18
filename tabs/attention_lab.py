import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.common import linear_projection, softmax, np_to_df, plot_heatmap

def render_attention_lab(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv):
    """어텐션 실험실 탭 렌더링"""
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
            fig = plot_heatmap(Ws, xticks=src_tokens, yticks=src_tokens, title="인코더 Self-Attention 가중치")
            st.pyplot(fig)
            plt.close(fig)  # 메모리 절약을 위해 그래프 자동 정리
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
            fig = plot_heatmap(Wd_self, xticks=tgt_tokens, yticks=tgt_tokens, title="디코더 마스크드 Self-Attention")
            st.pyplot(fig)
            plt.close(fig)  # 메모리 절약을 위해 그래프 자동 정리
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
            fig = plot_heatmap(Wd, xticks=src_tokens, yticks=tgt_tokens, title="크로스 어텐션 가중치 (타깃 행 / 소스 열)")
            st.pyplot(fig)
            plt.close(fig)  # 메모리 절약을 위해 그래프 자동 정리
            st.markdown("**히트맵 해석**:")
            st.markdown("- **행**: 영어 단어 (타깃)")
            st.markdown("- **열**: 한국어 단어 (소스)")
            st.markdown("- **색상**: 밝을수록 해당 영어 단어가 한국어 단어를 더 참고")
            st.markdown("- **예시**: 'I'가 '나는'을 참고, 'ate'가 '먹었어'를 참고")
    else:
        st.warning("먼저 '분석 시작' 버튼을 눌러주세요.")
