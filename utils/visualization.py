"""可视化工具"""

import os
from collections import defaultdict


def plot_summary(summary, title, save_path="outputs/summary.png"):
    """绘制算法对比图"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    # 配置中文字体
    import matplotlib.font_manager as fm
    fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
    available = [f.name for f in fm.fontManager.ttflist]
    for font in fonts:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font]
            break
    plt.rcParams['axes.unicode_minus'] = False

    # 提取数据
    algorithms = [s['algorithm'] for s in summary]
    objectives = [s['objective_value'] for s in summary]
    on_time_rates = [s['on_time_rate'] * 100 for s in summary]
    run_times = [s['run_time'] for s in summary]

    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # 目标函数
    axes[0].bar(algorithms, objectives, color=colors[:len(algorithms)])
    axes[0].set_ylabel('目标函数值', fontsize=11)
    axes[0].set_title('目标函数对比', fontsize=12)
    axes[0].tick_params(axis='x', rotation=15)
    for i, v in enumerate(objectives):
        axes[0].text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

    # 准时率
    axes[1].bar(algorithms, on_time_rates, color=colors[:len(algorithms)])
    axes[1].set_ylabel('准时率 (%)', fontsize=11)
    axes[1].set_title('准时率对比', fontsize=12)
    axes[1].set_ylim(0, 100)
    axes[1].axhline(y=80, color='r', linestyle='--', alpha=0.5, label='80%基准')
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].legend()
    for i, v in enumerate(on_time_rates):
        axes[1].text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

    # 运行时间
    axes[2].bar(algorithms, run_times, color=colors[:len(algorithms)])
    axes[2].set_ylabel('运行时间 (秒)', fontsize=11)
    axes[2].set_title('算法效率', fontsize=12)
    axes[2].tick_params(axis='x', rotation=15)
    for i, v in enumerate(run_times):
        axes[2].text(i, v, f'{v:.2f}s', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path


def plot_scalability_curve(all_records, title, save_path="outputs/scalability.png"):
    """绘制规模扩展性曲线"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    # 配置中文字体
    import matplotlib.font_manager as fm
    fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
    available = [f.name for f in fm.fontManager.ttflist]
    for font in fonts:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font]
            break
    plt.rcParams['axes.unicode_minus'] = False

    # 按算法和规模分组
    data_by_algo = defaultdict(lambda: defaultdict(list))
    for record in all_records:
        algo = record['algorithm']
        n = record['num_robots']
        data_by_algo[algo][n].append(record)

    scales = sorted(set(r['num_robots'] for r in all_records))

    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    colors = {'B&B': '#2ca02c', 'ALNS': '#1f77b4', 'Greedy-EDF': '#ff7f0e'}

    for algo_name, algo_data in data_by_algo.items():
        color = colors.get(algo_name, '#333333')

        x_vals = []
        obj_vals = []
        ontime_vals = []
        time_vals = []

        for n in scales:
            if n in algo_data:
                records = algo_data[n]
                x_vals.append(n)
                obj_vals.append(sum(r['objective_value'] for r in records) / len(records))
                ontime_vals.append(sum(r['on_time_rate'] for r in records) / len(records) * 100)
                time_vals.append(sum(r['run_time'] for r in records) / len(records))

        axes[0].plot(x_vals, obj_vals, marker='o', label=algo_name,
                    color=color, linewidth=2)
        axes[1].plot(x_vals, ontime_vals, marker='s', label=algo_name,
                    color=color, linewidth=2)
        axes[2].plot(x_vals, time_vals, marker='^', label=algo_name,
                    color=color, linewidth=2)

    axes[0].set_xlabel('机器人数量', fontsize=11)
    axes[0].set_ylabel('目标函数值', fontsize=11)
    axes[0].set_title('目标函数随规模变化', fontsize=12)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('机器人数量', fontsize=11)
    axes[1].set_ylabel('准时率 (%)', fontsize=11)
    axes[1].set_title('准时率随规模变化', fontsize=12)
    axes[1].set_ylim(0, 105)
    axes[1].axhline(y=80, color='r', linestyle='--', alpha=0.5)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('机器人数量', fontsize=11)
    axes[2].set_ylabel('运行时间 (秒)', fontsize=11)
    axes[2].set_title('算法效率随规模变化', fontsize=12)
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path