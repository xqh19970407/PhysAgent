import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# 设置随机种子保证可重复性
np.random.seed(42)

# 全局参数
g = 9.8  # 重力加速度（仅用于模拟）
mu = 0.0 # 摩擦系数
dt = 0.05  # 时间步长(s)
t_total = 4.0  # 总时长(s)
noise_level = 0.05  # 位移测量噪声标准差(m)

# 实验组设计
masses = np.array([0.2, 0.4, 0.6, 0.8, 1,2])  # 小车质量 (kg)
forces = np.array([0.1, 0.2, 0.3,0.4,0.5,1])  # 拉力 (N)

def simulate_motion_with_noise(F, m, t_total, dt):
    """模拟带噪声的小车运动数据"""
    t = np.arange(0, t_total, dt)
    
    # 真实物理过程（实验中未知）
    a_true = (F - mu * m * g) / m
    s_true = 0.5 * a_true * t**2
    
    # 添加噪声（模拟实际测量）
    position_noise = np.random.normal(0, noise_level, len(t))
    s_observed = s_true + position_noise
    
    # 确保位移始终增加（物理合理性检查）
    for i in range(1, len(s_observed)):
        if s_observed[i] < s_observed[i-1]:
            s_observed[i] = s_observed[i-1] + 1e-5  # 微小增量
    
    return pd.DataFrame({
        'time': t,
        'position': s_observed,
        'force': F,
        'mass': m,
        'a_true': a_true  # 仅用于验证，实际实验中不可见
    })

# 生成所有实验组数据
df_list = []
for m in masses:
    for F in forces:
        df_list.append(simulate_motion_with_noise(F, m, t_total, dt))
raw_data = pd.concat(df_list, ignore_index=True)
# 保存模拟数据
raw_data.to_csv('/home/xqhan/InvDesAgents/data/simulated_data.csv', index=False)
