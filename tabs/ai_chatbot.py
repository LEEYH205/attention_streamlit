import streamlit as st
import google.generativeai as genai

def render_ai_chatbot(analyze, src_tokens, tgt_tokens, src_E, tgt_E, gemini_model):
    """AI 챗봇 탭을 렌더링합니다."""
    
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
                        
                        # 채팅 히스토리에 저장
                        if 'chat_history' not in st.session_state:
                            st.session_state.chat_history = []
                        
                        st.session_state.chat_history.append({
                            'user': user_input,
                            'ai': response.text
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Gemini AI 응답 생성 실패: {str(e)}")
            else:
                st.warning("질문을 입력해주세요.")
        
        # 채팅 히스토리 표시
        if 'chat_history' in st.session_state and st.session_state.chat_history:
            st.markdown("### 📝 대화 히스토리")
            
            for i, chat in enumerate(st.session_state.chat_history):
                with st.expander(f"대화 {i+1}", expanded=False):
                    st.markdown(f"**👤 사용자:** {chat['user']}")
                    st.markdown(f"**🤖 Gemini AI:** {chat['ai']}")
            
            if st.button("🗑️ 대화 히스토리 삭제"):
                st.session_state.chat_history = []
                st.rerun()
        
        # 고급 기능
        st.markdown("### 🚀 고급 기능")
        
        # 어텐션 메커니즘 시뮬레이션
        if st.checkbox("어텐션 메커니즘 시뮬레이션"):
            st.markdown("**간단한 어텐션 계산 시뮬레이션:**")
            
            # 사용자 입력 문장
            user_sentence = st.text_input("분석할 문장을 입력하세요:", value="나는 머신러닝을 공부합니다")
            
            if user_sentence:
                tokens = user_sentence.split()
                st.write(f"**토큰화 결과:** {tokens}")
                
                # 간단한 어텐션 가중치 (예시)
                import numpy as np
                attention_weights = np.random.rand(len(tokens), len(tokens))
                attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
                
                st.write("**어텐션 가중치 (랜덤 예시):**")
                st.dataframe(attention_weights, 
                           index=[f"Query: {token}" for token in tokens],
                           columns=[f"Key: {token}" for token in tokens])
        
        # 어텐션 패턴 분석 도우미
        if st.checkbox("어텐션 패턴 분석 도우미"):
            st.markdown("**어텐션 패턴 분석을 위한 도구:**")
            
            pattern_type = st.selectbox(
                "분석할 어텐션 패턴 유형:",
                ["Self-Attention", "Cross-Attention", "Masked Attention"]
            )
            
            if st.button("패턴 분석 요청"):
                prompt = f"""
                {pattern_type}에 대해 다음을 설명해주세요:
                1. 기본 개념과 작동 원리
                2. 언제 사용되는지
                3. 장단점
                4. 실제 예시
                
                한국어로 간단명료하게 설명해주세요.
                """
                
                with st.spinner("패턴 분석 중..."):
                    try:
                        response = gemini_model.generate_content(prompt)
                        st.markdown(f"**🔍 {pattern_type} 분석 결과:**")
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"분석 실패: {str(e)}")
    
    else:
        st.warning("⚠️ Gemini API가 연결되지 않았습니다.")
        st.markdown("""
        **API 키 설정 방법:**
        1. 사이드바에 Google Gemini API 키를 입력하세요
        2. 또는 `config.env` 파일에 `GOOGLE_API_KEY`를 설정하세요
        3. API 키는 [Google AI Studio](https://aistudio.google.com/app/apikey)에서 발급받을 수 있습니다
        
        **대안:**
        - 기본 키워드 기반 챗봇을 사용할 수 있습니다
        - 또는 다른 AI 서비스를 연결할 수 있습니다
        """)
        
        # 기본 키워드 기반 챗봇 (fallback)
        st.markdown("### 🔍 기본 키워드 기반 챗봇")
        
        user_input = st.text_input("키워드를 입력하세요 (예: attention, transformer, softmax):")
        
        if user_input:
            # 간단한 키워드 기반 응답
            responses = {
                "attention": "어텐션 메커니즘은 입력 시퀀스의 특정 부분에 집중하는 방법입니다.",
                "transformer": "Transformer는 어텐션 메커니즘을 기반으로 한 신경망 아키텍처입니다.",
                "softmax": "Softmax는 벡터를 확률 분포로 변환하는 함수입니다.",
                "qkv": "Q(Query), K(Key), V(Value)는 어텐션 계산의 핵심 요소입니다.",
                "multihead": "멀티헤드 어텐션은 여러 개의 어텐션을 병렬로 수행합니다."
            }
            
            response = responses.get(user_input.lower(), "해당 키워드에 대한 정보를 찾을 수 없습니다.")
            st.info(f"**답변:** {response}")
