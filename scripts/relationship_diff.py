import numpy as np
import matplotlib.pyplot as plt

# 原始数据
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

# 计算各组数据对应的 (yaw+pitch+roll)
sum_pose = yaw + pitch + roll

# 定义第一行数据作为基准
base_facesim = facesim[0]
base_pose = sum_pose[0]

# 计算每个数据点相对于第一行数据的差值
delta_facesim = facesim - base_facesim
delta_pose = sum_pose - base_pose

# 对差值数据进行线性回归： Δfacesim = slope * (Δpose)
A = np.vstack([delta_pose, np.ones(len(delta_pose))]).T
slope, intercept = np.linalg.lstsq(A, delta_facesim, rcond=None)[0]
# 理想情况下，intercept应接近于0（如果数据完全相对于第一行作差，则第一行的差值为0）
print("线性拟合的模型: Δfacesim = {:.4f} * (Δ(yaw+pitch+roll)) + {:.4f}".format(slope, intercept))

# 为了观察整体的数据趋势，也画出原始数据的点和拟合曲线
plt.figure(figsize=(8, 6))
plt.scatter(delta_pose, delta_facesim, c="blue", label="data point")
x_fit = np.linspace(delta_pose.min()-1, delta_pose.max()+1, 100)
plt.plot(x_fit, slope * x_fit + intercept, color="red", label="linear")
plt.xlabel("Δ(yaw + pitch + roll)")
plt.ylabel("Δfacesim")
plt.title("relationship diff")
plt.legend()
plt.grid(True)
plt.show()