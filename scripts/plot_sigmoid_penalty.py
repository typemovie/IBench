import numpy as np
import matplotlib.pyplot as plt

def sigmoid_penalty(landmark_diff, T=1.0, alpha=10.0):
    """
    使用 Sigmoid 函数作为调控函数，并假设 landmark_diff 已归一化到 [0, 1] 范围。
    
    参数:
      landmark_diff: 已归一化的关键点差异值
      T: 理想的归一化目标（固定为 1.0）
      alpha: 控制函数陡峭度的单个超参数，数值越大，过渡越陡峭
      
    返回:
      Penalty 值
    """
    return 1 / (1 + np.exp(alpha * (landmark_diff - T)))

def main():
    # 参数设置: 假设 T 固定为 1.0, 仅调整 alpha
    T = 1.0
    alpha = 10.0

    # 生成归一化后的 landmark_diff 范围值 (0 到 1)
    x = np.linspace(0, 1, 300)
    y = sigmoid_penalty(x, T, alpha)
    
    # 绘制函数图
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=f'Sigmoid Penalty (T={T}, alpha={alpha})', linewidth=2)
    plt.xlabel('Normalized landmark_diff')
    plt.ylabel('Penalty')
    plt.title('Sigmoid Penalty Function')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.show()
    plt.savefig("sigmoid_penalty_function.png")
    
if __name__ == "__main__":
    main()