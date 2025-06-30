import numpy as np
import matplotlib.pyplot as plt
from math import exp

def compute_penalty(diff, lm_scale):
    """
    根据归一化关键点差异 diff 和调控参数 lm_scale 计算高斯调控函数值
    """
    return exp(- (diff / lm_scale) ** 2)

def plot_curve(lm_scale=0.2, diff_range=(0, 1), num_points=500):
    # 生成归一化关键点差异的取值范围
    diffs = np.linspace(diff_range[0], diff_range[1], num_points)
    # 计算每个 diff 对应的 penalty
    penalties = [compute_penalty(diff, lm_scale) for diff in diffs]
    
    plt.figure(figsize=(8, 5))
    plt.plot(diffs, penalties, label=f'lm_scale = {lm_scale}', color='blue')
    plt.title('高斯调控函数曲线')
    plt.xlabel('归一化关键点差异')
    plt.ylabel('penalty')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 你可以调整 lm_scale 参数或 diff_range 来观察不同情形下的曲线
    plot_curve(lm_scale=0.2)