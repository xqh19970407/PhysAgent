import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from matplotlib.animation import FuncAnimation

# 设置随机种子以确保可重现性
np.random.seed(42)

# 模拟参数
G = 6.67430e-11  # 万有引力常数 (m³ kg⁻¹ s⁻²)
M_sun = 1.989e30  # 太阳质量 (kg)
AU = 1.496e11     # 天文单位 (m)
year_sec = 365.25 * 24 * 3600  # 一年的秒数
noise_level = 1e8  # 固定噪声水平 (m)
fontsize = 18
# 行星轨道参数（加入升交点经度和近点角）
planets = {
    "Mercury": {"a": 0.387*AU, "e": 0.2056, "inclination": 7.0, "omega": 29.12, "Omega": 48.33},
    "Venus": {"a": 0.723*AU, "e": 0.0068, "inclination": 3.4, "omega": 54.85, "Omega": 76.68},
    "Earth": {"a": AU, "e": 0.0167, "inclination": 0.0, "omega": 114.21, "Omega": 0.0},
    "Mars": {"a": 1.524*AU, "e": 0.0934, "inclination": 1.9, "omega": 286.50, "Omega": 49.56},
    "Jupiter": {"a": 5.204*AU, "e": 0.0489, "inclination": 1.3, "omega": 273.87, "Omega": 100.46},
}

# 生成行星运动数据
def simulate_planet_orbit(a, e, inclination, omega, Omega, num_points=500):
    """模拟行星轨道，包含完整轨道要素"""
    b = a * np.sqrt(1 - e**2)  # 半短轴
    T = 2 * np.pi * np.sqrt(a**3 / (G * M_sun))  # 轨道周期
    
    # 生成时间序列
    t = np.linspace(0, T, num_points)
    mean_anomaly = 2 * np.pi * t / T
    
    # 求解开普勒方程
    eccentric_anomaly = optimize.newton(
        lambda E: E - e * np.sin(E) - mean_anomaly, mean_anomaly, maxiter=100
    )
    
    # 轨道平面内的位置
    x_orb = a * (np.cos(eccentric_anomaly) - e)
    y_orb = b * np.sin(eccentric_anomaly)
    
    # 计算真近点角
    true_anomaly = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(eccentric_anomaly / 2),
        np.sqrt(1 - e) * np.cos(eccentric_anomaly / 2)
    )
    
    # 轨道平面内的距离
    r_orb = a * (1 - e * np.cos(eccentric_anomaly))
    
    # 转换为三维坐标（考虑倾角 i、升交点经度 Ω、近点角 ω）
    i, Omega, omega = np.radians(inclination), np.radians(Omega), np.radians(omega)
    x = r_orb * (np.cos(Omega) * np.cos(true_anomaly + omega) - 
                 np.sin(Omega) * np.sin(true_anomaly + omega) * np.cos(i))
    y = r_orb * (np.sin(Omega) * np.cos(true_anomaly + omega) + 
                 np.cos(Omega) * np.sin(true_anomaly + omega) * np.cos(i))
    z = r_orb * np.sin(true_anomaly + omega) * np.sin(i)
    
    # 添加固定噪声
    x += np.random.normal(0, noise_level, num_points)
    y += np.random.normal(0, noise_level, num_points)
    z += np.random.normal(0, noise_level, num_points)
    
    # 解析速度计算
    v_x = -a * np.sin(eccentric_anomaly) * 2 * np.pi / T
    v_y = b * np.cos(eccentric_anomaly) * 2 * np.pi / T
    v = np.sqrt(v_x**2 + v_y**2)
    
    # 计算面积速度（验证开普勒第二定律）
    r_vec = np.array([x, y, z]).T
    v_vec = np.array([np.gradient(x, t), np.gradient(y, t), np.gradient(z, t)]).T
    cross_product = np.cross(r_vec, v_vec)
    areal_velocity = 0.5 * np.linalg.norm(cross_product, axis=1)
    
    return t, x, y, z, v, areal_velocity

# 模拟行星数据
planet_data = {}
for name, params in planets.items():
    t, x, y, z, v, areal_velocity = simulate_planet_orbit(
        params["a"], params["e"], params["inclination"], params["omega"], params["Omega"]
    )
    planet_data[name] = {
        "time": t,
        "x": x,
        "y": y,
        "z": z,
        "velocity": v,
        "areal_velocity": areal_velocity
    }
    # 验证面积定律
    print(f"{name} 的面积速度（应近似恒定）：均值 {np.mean(areal_velocity):.2e} ± 标准差 {np.std(areal_velocity):.2e}")


# 保存二进制数据
import pickle
with open("/home/xqhan/InvDesAgents/data/planet_data.pkl", "wb") as f:
    pickle.dump(planet_data, f)