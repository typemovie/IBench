import numpy as np
import matplotlib.pyplot as plt

# 数据
facesim = np.array([
    0.735393405, 0.643686354, 0.512071013, 0.635289431,
    0.61697799,  0.60079658,  0.574015498, 0.591027975,
    0.714251816, 0.713990152, 0.668103158, 0.361818492
])
yaw = np.array([
    9.298392704, 13.89054064, 27.4605236, 17.5905068,
    19.25169478, 20.51384262, 24.74528717, 23.09522479,
    11.23729566, 11.81353174, 14.45572858, 19.47661317
])
pitch = np.array([
    5.871601073, 7.100098231, 11.28290355, 8.288379935,
    9.033347594, 9.343295368, 10.41231312, 9.689472586,
    6.610124457, 6.722223327, 7.32767999, 10.0249493
])
roll = np.array([
    9.472624539, 10.72419407, 13.17417335, 11.45947604,
    11.89152007, 12.18473987, 13.25953731, 12.40500735,
    10.26232268, 10.60234477, 11.09901744, 13.21846886
])

# 计算face三参数之和
sum_pose = yaw + pitch + roll

# 以facesim为因变量、sum_pose为自变量建立线性回归模型
# 模型形式： facesim = A0 + B * (yaw+pitch+roll)
A = np.vstack([sum_pose, np.ones(len(sum_pose))]).T
B, A0 = np.linalg.lstsq(A, facesim, rcond=None)[0]
print("拟合的线性模型: facesim = {:.4f} * (yaw+pitch+roll) + {:.4f}".format(B, A0))

# 根据用户设想的参考值：利用第一组数据作为基准
# 第一组数据: sum_pose[0] ≈ 24.6426, facesim[0] ≈ 0.7354
k = (facesim[0] - 0.735393405) / (sum_pose[0] - 24.642618316) if sum_pose[0] != 24.642618316 else 0.00819
# 通过推导，可以得到另一种表达形式: facesim = 0.735393405 - 0.00819*( (yaw+pitch+roll) - 24.6426 )
# 与上面线性拟合得到的模型非常接近

# 绘制数据点及拟合曲线
plt.scatter(sum_pose, facesim, label="data point")
x_fit = np.linspace(sum_pose.min()-1, sum_pose.max()+1, 100)
plt.plot(x_fit, B*x_fit + A0, color="red", label="linear")
plt.xlabel("yaw + pitch + roll")
plt.ylabel("facesim")
plt.title("facesim with yaw+pitch+roll")
plt.legend()
plt.show()