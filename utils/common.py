import numpy as np
import pandas as pd
import streamlit as st

def softmax(x, axis=None):
    """수치적으로 안정적인 소프트맥스 함수"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def linear_projection(X, W):
    """선형 사영 함수"""
    return X @ W

def np_to_df(array, row_idx=None, col_idx=None):
    """NumPy 배열을 Pandas DataFrame으로 변환"""
    if row_idx is None:
        row_idx = [f"row_{i}" for i in range(array.shape[0])]
    if col_idx is None:
        col_idx = [f"col_{i}" for i in range(array.shape[1])]
    
    return pd.DataFrame(array, index=row_idx, columns=col_idx)

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

def plot_heatmap(weights, xticks=None, yticks=None, title="Attention Weights"):
    """어텐션 가중치 히트맵 플롯"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(weights, cmap='viridis')
    
    if xticks is not None:
        ax.set_xticks(np.arange(len(xticks)))
        ax.set_xticklabels(xticks, rotation=45, ha="right")
    
    if yticks is not None:
        ax.set_yticks(np.arange(len(yticks)))
        ax.set_yticklabels(yticks)
    
    ax.set_title(title)
    
    # 값 주석
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            value = weights[i, j]
            if value > 0.01:  # 의미있는 값만 표시
                ax.text(j, i, f'{value:.3f}', ha="center", va="center", 
                       color='white' if value > 0.3 else 'black', fontsize=9)
    
    plt.colorbar(im, ax=ax)
    return fig
