import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 模型类定义 ====================

class SolarFieldModel(torch.nn.Module):
    """太阳能集热器场模型实现"""

    def __init__(self, params):
        super(SolarFieldModel, self).__init__()
        self.A_sf = torch.tensor(params['A_sf'], dtype=torch.float32)
        self.rho = torch.tensor(params['rho'], dtype=torch.float32)
        self.c_p = torch.tensor(params['c_p'], dtype=torch.float32)
        self.beta = torch.tensor(params['beta'], dtype=torch.float32)
        self.H = torch.tensor(params['H'], dtype=torch.float32)
        self.L_a = torch.tensor(params['L_a'], dtype=torch.float32)
        self.n_cs = torch.tensor(params['n_cs'], dtype=torch.int32)
        self.L_eq = self.L_a * self.n_cs
        self.c1 = torch.tensor(params['c1'], dtype=torch.float32)

    def forward(self, T1, T2, F1, I, T_a):
        m_1 = (F1 * self.rho) / self.c1
        T_avg = (T1 + T2) / 2
        dT2_dt = (self.beta * I - (self.H / self.L_eq) * (T_avg - T_a) -
                  (self.c_p * m_1 * (T2 - T1)) / self.L_eq)
        dT2_dt = dT2_dt / (self.A_sf * self.rho * self.c_p)
        return dT2_dt

    def simulate(self, T1, F1, T_a, I, T2, dt=60):
        dT2_dt = self.forward(T1, T2, F1, I, T_a)
        T2 += dT2_dt * dt
        return T2


class StorageTankModel(torch.nn.Module):
    """储热罐双节点分层模型"""

    def __init__(self, params):
        super(StorageTankModel, self).__init__()
        self.rho = torch.tensor(params['rho'], dtype=torch.float32)
        self.TV = torch.tensor(params['TV'], dtype=torch.float32)
        self.c_p = torch.tensor(params['c_p'], dtype=torch.float32)
        self.UA1 = torch.tensor(params['UA1'], dtype=torch.float32)
        self.UA2 = torch.tensor(params['UA2'], dtype=torch.float32)

    def forward(self, T2, T3, T4, T6, T_a, m_1_zong, m_2):
        dT3_dt = (m_1_zong * T2 + m_2 * T4 - m_1_zong * T3 - m_2 * T3 -
                  self.UA1 * (T3 - T_a) / self.c_p) / (self.rho * self.TV)
        dT4_dt = (m_1_zong * T3 + m_2 * T6 - m_1_zong * T4 - m_2 * T4 -
                  self.UA2 * (T4 - T_a) / self.c_p) / (self.rho * self.TV)
        return dT3_dt, dT4_dt

    def simulate(self, T2, T6, T3, T4, T_a, m_1_zong, m_2, dt=60):
        dT3_dt, dT4_dt = self.forward(T2, T3, T4, T6, T_a, m_1_zong, m_2)
        T3 += dT3_dt * dt
        T4 += dT4_dt * dt
        return T3, T4


class HeatExchangerModel(torch.nn.Module):
    """换热器静态模型"""

    def __init__(self, params):
        super(HeatExchangerModel, self).__init__()
        self.A_he = torch.tensor(params['A_he'], dtype=torch.float32)
        self.alpha_he = torch.tensor(params['alpha_he'], dtype=torch.float32)
        self.c_p = torch.tensor(params['c_p'], dtype=torch.float32)
        self.c_p_sw = torch.tensor(params['c_p_sw'], dtype=torch.float32)
        self.rho = torch.tensor(params['rho'], dtype=torch.float32)
        self.rho_sw = torch.tensor(params['rho_sw'], dtype=torch.float32)

    def forward(self, T5, T7, m3, m4):
        theta_he = self.alpha_he * self.A_he * (1 / (m3 * self.c_p) - 1 / (m4 * self.c_p_sw))
        A = 1 - torch.exp(theta_he)
        B = ((m3 * self.c_p) / (m4 * self.c_p_sw)) * torch.exp(theta_he)
        eta_he1 = A / (1.1 - B)
        eta_he2 = (m3 * self.c_p) / (m4 * self.c_p_sw)

        T6_m = T5 - 1.6 * eta_he1 * (T5 - T7)
        T8_m = T7 + 0.45 * eta_he2 * (T5 - T6_m)
        return T6_m, T8_m


class SMDSystemModel(torch.nn.Module):
    """完整的SMD系统模型集成"""

    def __init__(self, params):
        super(SMDSystemModel, self).__init__()
        self.solar_field = SolarFieldModel(params)
        self.storage_tank = StorageTankModel(params)
        self.heat_exchanger = HeatExchangerModel(params)

    def simulate_full_system(self, V_sequence, I_sequence, T_a_sequence, F1_sequence, F2_sequence, F3_sequence,
                             dt=60, simulation_time=3600):
        # 初始化存储列表
        T2_history = []
        T3_history, T4_history = [], []
        T5_history, T6_history = [], []
        T7_history, T8_history = [], []

        # 初始条件
        T1 = torch.tensor(50.0)
        T2, T3 = torch.tensor(60.0), torch.tensor(60.0)
        T4 = torch.tensor(50.0)
        T5, T6 = torch.tensor(60.0), torch.tensor(25.0)
        T7, T8 = torch.tensor(50.0), torch.tensor(60.0)

        # 模拟主循环
        total_steps = simulation_time // dt
        for i in range(total_steps):
            if i < len(T_a_sequence):
                T_a = torch.tensor(T_a_sequence[i], dtype=torch.float32)

                # 获取当步的输入值
                F1_val = F1_sequence[i]
                V_val = V_sequence[i]
                I_val = I_sequence[i]

                # 转为 Tensor
                F1 = torch.tensor(F1_val, dtype=torch.float32)
                V = torch.tensor(V_val, dtype=torch.float32)
                I = torch.tensor(I_val, dtype=torch.float32)

                F2 = torch.tensor(F2_sequence[i], dtype=torch.float32)
                F3 = torch.tensor(F3_sequence[i], dtype=torch.float32)

            # 1. 太阳能场模拟
            T2 = self.solar_field.simulate(T1, F1, T_a, I, T2, dt)
            T2_history.append(T2.item())

            # 2. 定义质量流量
            m_1_zong = (F1 * self.solar_field.rho) / 6e4
            m_2 = (1 - V) * (F2 * self.solar_field.rho) / 6e4
            m_3 = (F2 * self.solar_field.rho) / 6e4
            m_4 = (F3 * self.heat_exchanger.rho_sw) / 6e4

            # 3. 储热罐模拟
            T3, T4 = self.storage_tank.simulate(T2, T6, T3, T4, T_a, m_1_zong, m_2, dt)
            T3_history.append(T3.item())
            T4_history.append(T4.item())
            T1 = T4.item()

            # 4. 计算混合温度
            T5 = V_val * T6 + (1 - V_val) * T3
            T7 = T8 - 4

            # 5. 热交换器模拟
            T6, T8 = self.heat_exchanger.forward(T5, T7, m_3, m_4)

            # 保存数据
            T5_history.append(T5.item() if isinstance(T5, torch.Tensor) else T5)
            T6_history.append(T6.item())
            T7_history.append(T7.item() if isinstance(T7, torch.Tensor) else T7)
            T8_history.append(T8.item())

        return {
            'T2': T2_history,
            'T3': T3_history,
            'T4': T4_history,
            'T5': T5_history,
            'T6': T6_history,
            'T7': T7_history,
            'T8': T8_history,
            'I_Input': I_sequence,
            'F1_Input': F1_sequence,
            'V_Input': V_sequence
        }


# ==================== 辅助函数：生成阶跃信号 ====================

def generate_step_sequence(min_val, max_val, total_steps, min_interval=20, max_interval=20):
    """生成随机阶跃信号"""
    sequence = []
    current_step = 0
    while current_step < total_steps:
        duration = random.randint(min_interval, max_interval)
        value = random.uniform(min_val, max_val)
        remaining = total_steps - current_step
        steps_to_fill = min(duration, remaining)
        sequence.extend([value] * steps_to_fill)
        current_step += steps_to_fill
    return sequence


# ==================== 主程序运行 ====================

model_params = {
    'A_sf': 0.007,
    'rho': 1000.0,
    'rho_sw': 1025.0,
    'c_p': 4200.0,
    'beta': 0.11,
    'H': 5.88,
    'L_a': 1.95,
    'n_cs': 5,
    'c1': 108e4,
    'TV': 1.5,
    'UA1': 1.6,
    'UA2': 8.8,
    'A_he': 1.65,
    'alpha_he': 670.80,
    'c_p_sw': 3900.0,
}

df_input = pd.read_csv("C:\huiwei\Documents\Python Scripts\input.csv")

T_a_sequence = df_input['T_a'].tolist()
I_sequence = df_input['I'].tolist()

if __name__ == "__main__":
    # 1. 设置模拟参数
    simulation_time = 3600  # 360分钟
    dt = 60  # 1分钟
    steps = simulation_time // dt  # 800个点

    # 固定环境参数

    F2_sequence = [10] * steps
    F3_sequence = [10] * steps

    # ==================== 输入生成 ====================


    F1_sequence = generate_step_sequence(8, 25, steps, min_interval=20, max_interval=20)

    V_sequence = generate_step_sequence(0, 0.8, steps, min_interval=20, max_interval=20)

    # ==================== 运行模拟 ====================

    smd_system = SMDSystemModel(model_params)
    results = smd_system.simulate_full_system(
        V_sequence, I_sequence, T_a_sequence, F1_sequence, F2_sequence, F3_sequence, dt, simulation_time
    )

    # ==================== 数据处理与保存 ====================
    df = pd.DataFrame(results)
    df['Time_Min'] = np.arange(0, simulation_time, dt) / 60

    # 采样间隔 = 1分钟
    sample_interval = 1
    df_sampled = df.iloc[::sample_interval, :].copy()

    # 选择需要保存的列
    cols_to_save = ['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'F1_Input', 'V_Input']
    df_final = df_sampled[cols_to_save]


    raw_filename = 'raw_data.csv'
    df_final.to_csv(raw_filename, index=False)
    print(f"原始数据已保存为: {raw_filename}")


    # Min-Max (x - min) / (max - min)
    df_normalized = (df_final - df_final.min()) / (df_final.max() - df_final.min())
    df_normalized = df_normalized.fillna(0)  #

    norm_filename = 'normalized_data.csv'
    df_normalized.to_csv(norm_filename, index=False)
    print(f"归一化数据已保存为: {norm_filename}")

    # ========================================
    plt.figure(figsize=(12, 12))

    plt.subplot(4, 1, 1)
    plt.plot(df['Time_Min'], df['I_Input'], 'orange', label='I (Irradiance)')
    plt.ylabel('Irradiance (W/m²)')
    plt.title('Ramp Input: I (600-900-600)')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(df['Time_Min'], df['F1_Input'], 'b-', label='F1 (Flow)')
    plt.ylabel('F1 (L/min)')
    plt.title('Step Input: F1 (20 min)')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(df['Time_Min'], df['V_Input'], 'g-', label='V (Valve)')
    plt.ylabel('V Opening')
    plt.title('Step Input: V (20 min)')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(df['Time_Min'], df['T2'], 'r-', label='T2 Response')
    plt.ylabel('Temperature (°C)')
    plt.title('System Response T2')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
