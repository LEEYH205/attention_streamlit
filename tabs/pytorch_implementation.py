import streamlit as st
import torch
import math
from utils.common import np_to_df

def render_pytorch_implementation(analyze, src_tokens, tgt_tokens, src_E, tgt_E, Wq, Wk, Wv):
    """PyTorch 구현 탭을 렌더링합니다."""
    
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
        
        # 추가적인 PyTorch 기능들
        st.markdown("### 🔧 PyTorch 고급 기능")
        
        # 그래디언트 계산
        if st.checkbox("그래디언트 계산 활성화"):
            src_E_torch.requires_grad_(True)
            Wq_torch.requires_grad_(True)
            
            # 어텐션 계산 (그래디언트 추적)
            Q_torch_grad = torch.matmul(src_E_torch, Wq_torch)
            K_torch_grad = torch.matmul(src_E_torch, Wk_torch)
            V_torch_grad = torch.matmul(src_E_torch, Wv_torch)
            
            scores_torch_grad = torch.matmul(Q_torch_grad, K_torch_grad.transpose(-2, -1)) / math.sqrt(d_k)
            weights_torch_grad = torch.softmax(scores_torch_grad, dim=-1)
            context_torch_grad = torch.matmul(weights_torch_grad, V_torch_grad)
            
            # 손실 함수 (예시)
            target = torch.randn_like(context_torch_grad)
            loss = torch.nn.functional.mse_loss(context_torch_grad, target)
            
            # 그래디언트 계산
            loss.backward()
            
            st.write("**그래디언트 정보:**")
            st.write(f"- src_E 그래디언트 shape: {src_E_torch.grad.shape}")
            st.write(f"- Wq 그래디언트 shape: {Wq_torch.grad.shape}")
            st.write(f"- 손실값: {loss.item():.6f}")
        
        # 멀티헤드 어텐션 구현
        st.markdown("### 🧩 PyTorch 멀티헤드 어텐션")
        
        num_heads = st.slider("헤드 수", min_value=1, max_value=8, value=4)
        
        if num_heads > 1:
            head_dim = d_k // num_heads
            
            # 헤드별로 분할
            Q_heads = Q_torch.view(Q_torch.size(0), num_heads, head_dim)
            K_heads = K_torch.view(K_torch.size(0), num_heads, head_dim)
            V_heads = V_torch.view(V_torch.size(0), num_heads, head_dim)
            
            # 각 헤드별 어텐션 계산
            head_outputs = []
            for i in range(num_heads):
                head_Q = Q_heads[:, i, :]
                head_K = K_heads[:, i, :]
                head_V = V_heads[:, i, :]
                
                head_scores = torch.matmul(head_Q, head_K.transpose(-2, -1)) / math.sqrt(head_dim)
                head_weights = torch.softmax(head_scores, dim=-1)
                head_context = torch.matmul(head_weights, head_V)
                head_outputs.append(head_context)
            
            # 헤드 결과 결합
            multihead_output = torch.cat(head_outputs, dim=-1)
            
            st.write(f"**멀티헤드 어텐션 결과:**")
            st.write(f"- 입력 shape: {Q_torch.shape}")
            st.write(f"- 헤드 수: {num_heads}")
            st.write(f"- 헤드 차원: {head_dim}")
            st.write(f"- 출력 shape: {multihead_output.shape}")
            
            # 헤드별 어텐션 가중치 시각화
            st.markdown("**헤드별 어텐션 가중치:**")
            for i in range(num_heads):
                head_weights = torch.softmax(
                    torch.matmul(Q_heads[:, i, :], K_heads[:, i, :].transpose(-2, -1)) / math.sqrt(head_dim),
                    dim=-1
                )
                st.write(f"Head {i+1} weights shape: {head_weights.shape}")
        
        # 성능 비교
        st.markdown("### ⚡ 성능 비교")
        
        if st.button("NumPy vs PyTorch 성능 비교"):
            import time
            import numpy as np
            
            # NumPy 구현
            start_time = time.time()
            Q_np = src_E @ Wq
            K_np = src_E @ Wk
            V_np = src_E @ Wv
            scores_np = Q_np @ K_np.T / np.sqrt(d_k)
            weights_np = np.exp(scores_np - np.max(scores_np, axis=-1, keepdims=True))
            weights_np = weights_np / np.sum(weights_np, axis=-1, keepdims=True)
            context_np = weights_np @ V_np
            numpy_time = time.time() - start_time
            
            # PyTorch 구현
            start_time = time.time()
            Q_torch = torch.matmul(src_E_torch, Wq_torch)
            K_torch = torch.matmul(src_E_torch, Wk_torch)
            V_torch = torch.matmul(src_E_torch, Wv_torch)
            scores_torch = torch.matmul(Q_torch, K_torch.transpose(-2, -1)) / math.sqrt(d_k)
            weights_torch = torch.softmax(scores_torch, dim=-1)
            context_torch = torch.matmul(weights_torch, V_torch)
            pytorch_time = time.time() - start_time
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("NumPy 실행 시간", f"{numpy_time:.6f}초")
            with col2:
                st.metric("PyTorch 실행 시간", f"{pytorch_time:.6f}초")
            
            if numpy_time > pytorch_time:
                st.success("🎉 PyTorch가 더 빠릅니다!")
            else:
                st.info("📊 NumPy가 더 빠릅니다.")
        
        # PyTorch 모델 저장/로드
        st.markdown("### 💾 모델 저장/로드")
        
        if st.button("PyTorch 모델 저장"):
            # 간단한 어텐션 모델 생성
            attention_model = torch.nn.MultiheadAttention(
                embed_dim=d_k,
                num_heads=num_heads,
                batch_first=True
            )
            
            # 모델 저장
            torch.save(attention_model.state_dict(), "attention_model.pth")
            st.success("✅ 모델이 'attention_model.pth'로 저장되었습니다!")
        
        if st.button("PyTorch 모델 로드"):
            try:
                # 모델 로드
                attention_model = torch.nn.MultiheadAttention(
                    embed_dim=d_k,
                    num_heads=num_heads,
                    batch_first=True
                )
                attention_model.load_state_dict(torch.load("attention_model.pth"))
                st.success("✅ 모델이 성공적으로 로드되었습니다!")
                
                # 모델 정보 표시
                st.write("**로드된 모델 정보:**")
                st.write(f"- 임베딩 차원: {d_k}")
                st.write(f"- 헤드 수: {num_heads}")
                st.write(f"- 모델 파라미터 수: {sum(p.numel() for p in attention_model.parameters())}")
                
            except FileNotFoundError:
                st.error("❌ 저장된 모델 파일을 찾을 수 없습니다. 먼저 모델을 저장해주세요.")
        
        # PyTorch 최적화 팁
        st.markdown("### 💡 PyTorch 최적화 팁")
        
        with st.expander("PyTorch 성능 최적화 방법"):
            st.markdown("""
            **1. 메모리 효율성:**
            - `torch.no_grad()` 사용하여 불필요한 그래디언트 계산 방지
            - 적절한 데이터 타입 선택 (float32 vs float16)
            
            **2. 연산 최적화:**
            - `torch.matmul` 대신 `@` 연산자 사용
            - 배치 처리로 벡터화 연산 활용
            
            **3. GPU 활용:**
            - `tensor.cuda()` 또는 `tensor.to('cuda')` 사용
            - `torch.cuda.empty_cache()`로 GPU 메모리 정리
            
            **4. 컴파일 최적화:**
            - PyTorch 2.0+ `torch.compile()` 사용
            - JIT 컴파일로 실행 속도 향상
            """)
    
    else:
        st.warning("먼저 '분석 시작' 버튼을 눌러주세요.")
