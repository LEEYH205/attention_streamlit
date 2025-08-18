import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def render_learning_guide():
    """학습 가이드 탭 렌더링"""
    st.subheader("🎯 어텐션 메커니즘 학습 가이드")
    st.markdown("""
    어텐션 메커니즘을 단계별로 학습해보세요. 각 단계를 클릭하면 해당 내용을 자세히 볼 수 있습니다.
    """)
    
    # 트랜스포머 전체 그림
    st.markdown("## 🏗️ 트랜스포머 아키텍처 한눈에 보기")
    st.markdown("""
    우리가 배울 어텐션은 트랜스포머 모델의 핵심 부품입니다. 아래 다이어그램은 번역 모델의 전체 구조를 보여줍니다. 
    각 부분이 어떻게 상호작용하는지 살펴보세요.
    """)
    
    # Plotly를 사용한 인터랙티브 트랜스포머 구조
    # 트랜스포머 구조 시각화
    fig = go.Figure()
    
    # 인코더 영역
    fig.add_trace(go.Scatter(
        x=[0, 0, 0, 0], y=[0, 1, 2, 3],
        mode='markers+text',
        marker=dict(size=20, color='lightblue'),
        text=['입력<br>(한국어)', 'Self-Attention', 'Feed Forward', '출력'],
        textposition="middle right",
        name="인코더",
        hovertemplate="<b>%{text}</b><extra></extra>"
    ))
    
    # 디코더 영역
    fig.add_trace(go.Scatter(
        x=[2, 2, 2, 2, 2], y=[0, 1, 2, 3, 4],
        mode='markers+text',
        marker=dict(size=20, color='lightcoral'),
        text=['출력<br>(영어)', 'Masked<br>Self-Attention', 'Cross-Attention', 'Feed Forward', '번역'],
        textposition="middle left",
        name="디코더",
        hovertemplate="<b>%{text}</b><extra></extra>"
    ))
    
    # 연결선 (Cross-Attention) - 더 자연스러운 곡선으로
    fig.add_trace(go.Scatter(
        x=[0.3, 0.8, 1.7], y=[3.2, 2.5, 2.2],
        mode='lines',
        line=dict(width=4, color='green', dash='dash', shape='spline'),
        name="K, V 전달",
        showlegend=True,
        hoverinfo='text',
        text=['인코더 출력', '중간 경로', '디코더 Cross-Attention']
    ))
    
    # 정보 흐름 화살표 - 더 자연스러운 위치로 조정
    # 인코더 내부 (파란색 화살표)
    encoder_arrows = [
        (0.15, 0.5, 0.25, 1.5),    # 입력 → Self-Attention
        (0.15, 1.5, 0.25, 2.5),    # Self-Attention → Feed Forward  
        (0.15, 2.5, 0.25, 3.5)     # Feed Forward → 출력
    ]
    
    for x1, y1, x2, y2 in encoder_arrows:
        fig.add_annotation(
            x=x1, y=y1,
            xref="x", yref="y",
            ax=x2, ay=y2,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=3,
            arrowcolor="blue",
            showarrow=True
        )
    
    # 디코더 내부 (빨간색 화살표)
    decoder_arrows = [
        (1.85, 0.5, 1.75, 1.5),    # 출력 → Masked Self-Attention
        (1.85, 1.5, 1.75, 2.5),    # Masked Self-Attention → Cross-Attention
        (1.85, 2.5, 1.75, 3.5),    # Cross-Attention → Feed Forward
        (1.85, 3.5, 1.75, 4.5)     # Feed Forward → 번역
    ]
    
    for x1, y1, x2, y2 in decoder_arrows:
        fig.add_annotation(
            x=x1, y=y1,
            xref="x", yref="y",
            ax=x2, ay=y2,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=3,
            arrowcolor="red",
            showarrow=True
        )
    
    # 레이아웃 설정
    fig.update_layout(
        title="트랜스포머 번역 모델 구조",
        xaxis=dict(range=[-0.5, 2.5], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.5, 4.5], showgrid=False, zeroline=False, showticklabels=False),
        width=800,
        height=500,
        showlegend=True,
        legend=dict(x=0.5, y=1.02, xanchor="center", yanchor="bottom", orientation="h")
    )
    
    # 배경에 박스 추가
    fig.add_shape(
        type="rect", x0=-0.3, y0=-0.3, x1=0.3, y1=3.3,
        fillcolor="lightblue", opacity=0.2, line=dict(color="blue", width=2)
    )
    fig.add_shape(
        type="rect", x0=1.7, y0=-0.3, x1=2.3, y1=4.3,
        fillcolor="lightcoral", opacity=0.2, line=dict(color="red", width=2)
    )
    
    # 박스 라벨
    fig.add_annotation(x=0, y=3.5, text="인코더 (한국어 처리)", showarrow=False, font=dict(size=14, color="blue"))
    fig.add_annotation(x=2, y=4.5, text="디코더 (영어 생성)", showarrow=False, font=dict(size=14, color="red"))
    
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("🔍 다이어그램 해설"):
        st.markdown("""
        **인코더 (왼쪽, 파란색):**
        - **입력**: 한국어 문장 "나는 밥을 먹었어"
        - **Self-Attention**: 문장 내 단어들 간의 관계 학습
        - **Feed Forward**: 각 단어의 표현을 더 풍부하게 만듦
        
        **디코더 (오른쪽, 빨간색):**
        - **출력**: 영어 문장 "I ate a meal" 생성
        - **Masked Self-Attention**: 이미 생성된 단어들만 참고 (미래 단어 차단)
        - **Cross-Attention**: 한국어 문장의 어떤 부분을 참고할지 결정
        - **Feed Forward**: 최종 영어 단어 예측
        
        **연결선 (초록색 점선):**
        - 인코더의 Key, Value 정보를 디코더의 Cross-Attention에 전달
        - **이것이 바로 우리가 학습할 Cross-Attention입니다!**
        """)
    
    st.markdown("---")
    
    # 단계별 학습 가이드
    step = st.selectbox("학습 단계 선택", 
                        ["0. 개요", "1. 토큰화 & 임베딩", "2. 선형사영(Q,K,V)", 
                         "3. 유사도(QKᵀ)", "4. 스케일링(/√dₖ)", "5. 소프트맥스(가중치)", "6. 가중합(컨텍스트)"])
    
    if step == "0. 개요":
        st.markdown("## 🎯 어텐션 메커니즘 개요")
        st.markdown("""
        **어텐션(Attention)**은 딥러닝에서 중요한 정보에 집중하는 메커니즘입니다.
        
        ### 🔍 핵심 아이디어
        - **질문(Query)**: "무엇을 찾고 있는가?"
        - **검색키(Key)**: "어떤 정보가 있는가?"
        - **정보(Value)**: "실제 내용은 무엇인가?"
        
        ### 📚 비유로 이해하기
        **도서관에서 책 찾기**와 같습니다:
        - **Query**: "밥 먹었다는 내용의 영어 표현을 찾고 있어"
        - **Key**: 책의 목차나 키워드
        - **Value**: 책의 실제 내용
        - **Attention**: 관련성 높은 책을 더 오래 읽기
        """)
        
    elif step == "1. 토큰화 & 임베딩":
        st.markdown("## 🔤 1단계: 토큰화 & 임베딩")
        st.markdown("""
        ### 📝 토큰화 (Tokenization)
        문장을 의미 있는 단위로 나누는 과정입니다.
        
        **예시**: "나는 밥을 먹었어" → ["나는", "밥을", "먹었어"]
        
        ### 🎯 임베딩 (Embedding)
        각 토큰을 고차원 벡터로 변환하는 과정입니다.
        
        **수식**: `E = Embedding(Tokens)`
        
        **의미**: 
        - 각 단어가 의미적으로 유사한 단어들과 가까운 벡터 공간에 위치
        - "먹었어"와 "먹다"는 비슷한 벡터를 가짐
        """)
        
    elif step == "2. 선형사영(Q,K,V)":
        st.markdown("## 🔄 2단계: 선형사영(Q,K,V)")
        st.markdown("""
        ### 🎯 Query, Key, Value 생성
        임베딩을 세 가지 다른 역할로 변환합니다.
        
        **수식**:
        - `Q = E × W_Q` (Query: 찾고자 하는 정보)
        - `K = E × W_K` (Key: 검색 키워드)
        - `V = E × W_V` (Value: 실제 정보)
        
        ### 💡 각각의 역할
        - **Query**: "무엇을 찾고 있는가?" (예: "먹었어"의 의미)
        - **Key**: "어떤 정보가 있는가?" (예: "먹다" 동사 관련)
        - **Value**: "실제 내용은 무엇인가?" (예: 과거형, 완료형 등)
        """)
        
    elif step == "3. 유사도(QKᵀ)":
        st.markdown("## 🔗 3단계: 유사도(QKᵀ)")
        st.markdown("""
        ### 📊 유사도 계산
        Query와 Key 간의 유사도를 계산합니다.
        
        **수식**: `Scores = Q × K^T`
        
        ### 🎯 의미
        - **높은 값**: Query와 Key가 매우 유사함
        - **낮은 값**: Query와 Key가 거의 관련 없음
        
        ### 💡 예시
        - "먹었어"(Query)와 "먹다"(Key) → 높은 유사도
        - "먹었어"(Query)와 "달다"(Key) → 낮은 유사도
        """)
        
    elif step == "4. 스케일링(/√dₖ)":
        st.markdown("## 📏 4단계: 스케일링(/√dₖ)")
        st.markdown("""
        ### 🌡️ 온도 조절
        유사도 점수를 적절한 범위로 조정합니다.
        
        **수식**: `Scores_scaled = Scores / √d_k`
        
        ### 🎯 왜 필요한가?
        - **고차원 문제**: 차원이 클수록 내적 값이 커짐
        - **소프트맥스 포화**: 너무 큰 값은 소프트맥스에서 0이나 1에 가까워짐
        - **학습 안정성**: 적절한 범위에서 학습이 안정적으로 진행됨
        
        ### 💡 비유
        **온도 조절**과 같습니다:
        - 너무 뜨거우면 → 모든 것이 타버림 (극단적 값)
        - 너무 차갑으면 → 아무것도 녹지 않음 (변화 없음)
        - 적당한 온도 → 적절한 변화 (안정적 학습)
        """)
        
    elif step == "5. 소프트맥스(가중치)":
        st.markdown("## 🎲 5단계: 소프트맥스(가중치)")
        st.markdown("""
        ### 📊 확률 분포로 변환
        유사도 점수를 확률 분포로 변환합니다.
        
        **수식**: `Attention_Weights = Softmax(Scores_scaled)`
        
        ### 🎯 소프트맥스의 역할
        - **정규화**: 모든 가중치의 합이 1이 됨
        - **비교**: 상대적인 중요도를 나타냄
        - **안정성**: 수치적으로 안정적인 계산
        
        ### 🔬 수치적 안정성
        **문제**: 지수 함수에서 큰 값이 오버플로우 발생
        **해결**: `exp(x - max(x))` 형태로 최대값을 빼줌
        
        **예시**:
        ```python
        # 안전하지 않은 방법
        exp(1000) → 오버플로우!
        
        # 안전한 방법
        exp(1000 - 1000) = exp(0) = 1 ✅
        ```
        """)
        
    elif step == "6. 가중합(컨텍스트)":
        st.markdown("## 🎯 6단계: 가중합(컨텍스트)")
        st.markdown("""
        ### 🔗 최종 출력 계산
        어텐션 가중치를 사용하여 Value의 가중 평균을 계산합니다.
        
        **수식**: `Context = Attention_Weights × Value`
        
        ### 🎯 의미
        - **가중 평균**: 중요한 정보에 더 집중
        - **문맥 정보**: 전체 문장의 맥락을 반영
        - **새로운 표현**: 원래 임베딩보다 풍부한 정보
        
        ### 💡 예시
        **"먹었어"**의 컨텍스트 벡터:
        - "먹었어" 자체: 0.6 (자기 자신)
        - "밥을": 0.3 (먹는 대상)
        - "나는": 0.1 (행위자)
        
        **결과**: "먹었어"가 "밥을 먹는 행동"이라는 맥락을 가진 새로운 표현
        """)
        
        # 소프트맥스 수치적 안정성 데모
        st.markdown("### 🔬 소프트맥스 수치적 안정성 데모")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**안전하지 않은 방법**")
            x = np.array([1000, 1001, 1002])
            try:
                exp_x = np.exp(x)
                softmax_unsafe = exp_x / np.sum(exp_x)
                st.write("결과:", softmax_unsafe)
                st.error("오버플로우 발생!")
            except:
                st.error("계산 불가능!")
        
        with col2:
            st.markdown("**안전한 방법**")
            x = np.array([1000, 1001, 1002])
            x_max = np.max(x)
            exp_x_safe = np.exp(x - x_max)
            softmax_safe = exp_x_safe / np.sum(exp_x_safe)
            st.write("결과:", softmax_safe)
            st.success("정상 계산!")
        
        st.markdown("**해석**: 안전한 방법은 모든 값에서 동일한 결과를 얻습니다.")
