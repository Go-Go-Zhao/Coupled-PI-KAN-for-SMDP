import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号'-'显示为方块的问题

# 设置随机种子以保证结果可重现（可选）
np.random.seed(6)

# 参数设置
initial_value = 0.1  # 初始值
min_value = 0  # 最小值
max_value = 0.8  # 最大值
step_duration = 24  # 每个阶梯持续时间（分钟）
num_steps = 15  # 阶梯数量（总共变化50次）
total_time = num_steps * step_duration  # 总时间

# 生成时间序列（每秒一个数据点）
time = np.arange(0, total_time, 1)

# 初始化数据数组
data = np.zeros_like(time, dtype=float)

# 记录每次变化的信息
step_info = []

# 生成阶梯变化
current_value = initial_value
current_time_index = 0

print("开始生成50次阶梯变化...")

for step in range(num_steps):
    # 生成随机阶跃值（1、2、3中随机选择）
    step_change = np.random.choice([0.1, 0.2, 0.3, 0.4])

    # 随机决定是增加还是减少（为了在范围内变化）
    # 这样可以确保在8-15范围内有更多的变化可能性
    direction = np.random.choice([-1, 1])
    step_change = step_change * direction

    # 计算新值
    new_value = current_value + step_change

    # 处理边界情况 - 确保值在8-15范围内
    if new_value > max_value:
        # 如果超过上限，调整为减少
        new_value = current_value - abs(step_change)
        if new_value < min_value:  # 双重保险
            new_value = min_value
    elif new_value < min_value:
        # 如果低于下限，调整为增加
        new_value = current_value + abs(step_change)
        if new_value > max_value:  # 双重保险
            new_value = max_value

    # 记录这次变化的信息
    step_info.append({
        'step_number': step + 1,
        'start_time': current_time_index,
        'end_time': current_time_index + step_duration - 1,
        'previous_value': current_value,
        'new_value': new_value,
        'change_amount': new_value - current_value,
        'duration': step_duration
    })

    # 应用阶跃变化
    end_index = min(current_time_index + step_duration, len(data))
    data[current_time_index:end_index] = new_value

    # 更新当前值和时间索引
    current_value = new_value
    current_time_index = end_index

    print(
        f"第{step + 1}次变化: {step_info[-1]['previous_value']} -> {new_value} (变化量: {step_info[-1]['change_amount']})")

# 创建DataFrame以便保存和分析
df = pd.DataFrame({
    'Time(min)': time,
    'Value': data
})

# 显示统计信息
print(f"\n数据统计:")
print(f"总数据点数: {len(data)}")
print(f"最小值: {data.min()}")
print(f"最大值: {data.max()}")
print(f"平均值: {data.mean():.2f}")
print(f"总共变化次数: {num_steps}")
print(f"每次变化持续时间: {step_duration}秒")
print(f"总模拟时间: {total_time}秒 ({total_time / 3600:.1f}小时)")

# 分析变化量的分布
changes = [info['change_amount'] for info in step_info]
unique_changes, counts = np.unique(changes, return_counts=True)
print(f"\n变化量统计:")
for change, count in zip(unique_changes, counts):
    print(f"  变化量 {change}: {count}次")

# 绘制阶梯图
plt.figure(figsize=(9, 6))
plt.step(time, data, where='post', linewidth=1.5)
plt.title(f'A stepped change(V1)', fontsize=16)
plt.xlabel('Time (min)', fontsize=16)
plt.ylabel('Value', fontsize=16)
plt.grid(True, alpha=0.3)
plt.ylim(min_value - 0.2, max_value + 0.2)

# 添加水平参考线
plt.axhline(y=min_value, color='r', linestyle='--', alpha=0.7, label=f'min_value ({min_value})')
plt.axhline(y=max_value, color='g', linestyle='--', alpha=0.7, label=f'max_value ({max_value})')
plt.legend(loc='upper right', fontsize=14)



# 添加一些统计信息到图上
plt.text(0.02, 0.97, f'num_steps: {num_steps}\nmean: {data.mean():.2f}\nstep_duration: {step_duration} min',
         transform=plt.gca().transAxes, verticalalignment='top', fontsize=13,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
# , pad=1.0
plt.tight_layout()
plt.show()

# 保存数据到CSV文件
# df.to_csv('step_input_data_50_changes.csv', index=False)
# print(f"\n数据已保存到 'step_input_data_50_changes.csv'")
#
# # 保存变化信息到另一个文件（可选）
# step_df = pd.DataFrame(step_info)
# step_df.to_csv('step_change_info.csv', index=False)
# print(f"变化详细信息已保存到 'step_change_info.csv'")