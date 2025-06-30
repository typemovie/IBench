import numpy as np
import matplotlib.pyplot as plt

def penalty_function(landmark_diff, T, sigma):
    """
    计算高斯调控函数 penalty。
    
    参数:
      landmark_diff: 关键点差异值
      T: 理想的关键点变化目标
      sigma: 标准差，用于控制高斯函数的宽度
      
    返回:
      高斯调控函数的值
    """
    return np.exp(-((landmark_diff - T) ** 2) / (2 * sigma ** 2))

def main():
    # 参数设置
    T = 0.2
    sigma = 0.1

    # 生成 landmark_diff 的范围值
    x = np.linspace(0, 30, 300)
    y = penalty_function(x, T, sigma)
    
    # 绘制函数图
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=f'Penalty (T={T}, sigma={sigma})', linewidth=2)
    plt.xlabel('landmark_diff')
    plt.ylabel('Penalty')
    plt.title('Gaussian Penalty Function')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 显示图像
    plt.show()
    
    # 保存图像到文件
    plt.savefig("penalty_function.png")
    
if __name__ == "__main__":
    main()