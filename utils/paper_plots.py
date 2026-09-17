"""论文图表绘制工具模块

生成高质量的学术论文图表，支持 PNG 和 PDF 双格式输出。
风格：兼顾学术严谨性和现代美观性。
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from collections import defaultdict

# 设置中文字体 - 检测可用字体
def setup_chinese_font():
    """配置中文字体，自动检测可用字体"""
    # 候选字体列表（按优先级）
    candidate_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 
                       'FangSong', 'STHeiti', 'STSong', 'Arial Unicode MS']
    
    # 获取系统可用字体
    available_fonts = set(f.name for f in fm.fontManager.ttflist)
    
    # 选择第一个可用的字体
    selected_font = None
    for font in candidate_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans']
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    # 确保负号正确显示（使用ASCII减号而不是Unicode减号）
    plt.rcParams['axes.unicode_minus'] = False
    # 设置mathtext使用常规字体
    plt.rcParams['mathtext.default'] = 'regular'
    return selected_font

# 初始化字体
_font = setup_chinese_font()

# 全局样式
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (10, 6)

# 现代配色方案
COLORS = {
    'B&B': '#2E86AB',      # 深蓝
    'ALNS': '#A23B72',     # 玫红
    'Greedy-EDF': '#F18F01',  # 橙色
    'Greedy': '#F18F01',
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'gray': '#6C757D',
}

# 算子名称映射
DESTROY_OP_NAMES = ['Random', 'Worst', 'Related']
REPAIR_OP_NAMES = ['Smart', 'Regret-2', 'Batch']


def save_figure(fig, filepath, formats=['png']):
    """保存图表为多种格式"""
    base_path = os.path.splitext(filepath)[0]
    for fmt in formats:
        output_path = f"{base_path}.{fmt}"
        fig.savefig(output_path, format=fmt, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"  [✓] 已保存: {output_path}")


def plot_performance_overview(phase1_data, phase2_data, output_dir):
    """
    Figure 1: 性能全景图
    Panel A: 小规模箱线图 (N=4,6)
    Panel B: 大规模折线图 (N=8~50)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # === Panel A: 小规模箱线图 ===
    ax1 = axes[0]
    
    # 按规模和算法分组
    small_scale_data = defaultdict(lambda: defaultdict(list))
    for result in phase1_data:
        n = result['num_tasks']
        algo = result['algorithm']
        obj = result['objective_value']
        small_scale_data[n][algo].append(obj)
    
    # 准备箱线图数据
    positions = []
    box_data = []
    colors_list = []
    labels = []
    
    x_pos = 0
    tick_positions = []
    tick_labels = []
    
    for n in sorted(small_scale_data.keys()):
        algos = ['B&B', 'ALNS', 'Greedy-EDF']
        start_pos = x_pos
        for algo in algos:
            if algo in small_scale_data[n] and small_scale_data[n][algo]:
                positions.append(x_pos)
                box_data.append(small_scale_data[n][algo])
                colors_list.append(COLORS.get(algo, COLORS['gray']))
                labels.append(algo)
                x_pos += 1
        tick_positions.append((start_pos + x_pos - 1) / 2)
        tick_labels.append(f'N={n}')
        x_pos += 0.5
    
    if box_data:
        bp = ax1.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                        showfliers=False)  # 隐藏异常值点，让图表更清晰
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for element in ['whiskers', 'caps', 'medians']:
            plt.setp(bp[element], color='#333333', linewidth=1.5)
    
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels)
    ax1.set_ylabel('目标函数值 (Objective Value)', fontsize=11)
    ax1.set_title('(A) 小规模最优性验证', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 图例
    legend_handles = [mpatches.Patch(color=COLORS[algo], alpha=0.7, label=algo) 
                     for algo in ['B&B', 'ALNS', 'Greedy-EDF']]
    ax1.legend(handles=legend_handles, loc='upper right')
    
    # === Panel B: 大规模折线图 ===
    ax2 = axes[1]
    
    # 按规模和算法聚合
    large_scale_data = defaultdict(lambda: defaultdict(list))
    for result in phase2_data:
        n = result['num_tasks']
        algo = result['algorithm']
        obj = result['objective_value']
        large_scale_data[n][algo].append(obj)
    
    scales = sorted(large_scale_data.keys())
    
    for algo in ['ALNS', 'Greedy-EDF']:
        means = []
        stds = []
        valid_scales = []
        for n in scales:
            if algo in large_scale_data[n] and large_scale_data[n][algo]:
                values = large_scale_data[n][algo]
                means.append(np.mean(values))
                stds.append(np.std(values))
                valid_scales.append(n)
        
        if means:
            means = np.array(means)
            stds = np.array(stds)
            color = COLORS.get(algo, COLORS['gray'])
            ax2.plot(valid_scales, means, 'o-', color=color, label=algo, 
                    linewidth=2, markersize=6)
            ax2.fill_between(valid_scales, means - stds, means + stds, 
                            color=color, alpha=0.2)
    
    ax2.set_xlabel('任务规模 N', fontsize=11)
    ax2.set_ylabel('目标函数值 (Objective Value)', fontsize=11)
    ax2.set_title('(B) 规模扩展性分析', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'fig1_performance_overview'))
    plt.close()


def plot_scalability(all_data, output_dir):
    """
    Figure 3: 计算效率与扩展性气泡图
    X轴：规模N，Y轴：运行时间(log)，气泡大小：迭代次数/节点数
    """
    from matplotlib.ticker import LogLocator, FuncFormatter
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 按算法分组
    algo_data = defaultdict(lambda: {'n': [], 'time': [], 'size': []})
    
    for result in all_data:
        algo = result['algorithm']
        n = result['num_tasks']
        run_time = result.get('run_time', 0.001)
        
        # 气泡大小：B&B用节点数，ALNS用迭代次数，Greedy固定小
        if algo == 'B&B':
            size = result.get('nodes_explored', 100)
        elif algo == 'ALNS':
            size = result.get('iterations', 300)
        else:
            size = 50
        
        algo_data[algo]['n'].append(n)
        algo_data[algo]['time'].append(max(run_time, 0.001))
        algo_data[algo]['size'].append(size)
    
    # 绘制气泡图
    for algo, data in algo_data.items():
        color = COLORS.get(algo, COLORS['gray'])
        sizes = np.array(data['size'])
        # 归一化气泡大小
        sizes = 50 + (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1) * 200
        
        ax.scatter(data['n'], data['time'], s=sizes, c=color, 
                  alpha=0.6, label=algo, edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('任务规模 N', fontsize=12)
    ax.set_ylabel('运行时间 (秒, Log Scale)', fontsize=12)
    ax.set_yscale('log')
    
    # 使用自定义格式化器避免Unicode减号
    def format_func(value, pos):
        if value >= 1:
            return f'{value:.0f}'
        elif value >= 0.01:
            return f'{value:.2f}'
        else:
            return f'{value:.3f}'
    
    ax.yaxis.set_major_formatter(FuncFormatter(format_func))
    
    ax.set_title('计算效率与扩展性分析', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    
    # 添加"计算墙"标注
    ax.axhline(y=120, color=COLORS['danger'], linestyle='--', alpha=0.7)
    ax.text(max([max(d['n']) for d in algo_data.values()]) - 5, 140, 
            '时间限制 (120s)', color=COLORS['danger'], fontsize=10)
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'fig3_scalability'))
    plt.close()


def plot_convergence(iteration_history, output_dir, title_suffix=''):
    """
    Figure 4A: ALNS收敛轨迹图
    双线：当前解（波动）vs 历史最优解（阶梯下降）
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    iterations = iteration_history['iterations']
    current_cost = iteration_history['current_cost']
    best_cost = iteration_history['best_cost']
    
    ax.plot(iterations, current_cost, color=COLORS['gray'], alpha=0.5, 
            linewidth=1, label='当前解 (Current)')
    ax.plot(iterations, best_cost, color=COLORS['ALNS'], linewidth=2, 
            label='历史最优解 (Best)')
    
    # 标记改进点
    improved = iteration_history['improved']
    improve_iters = [i for i, imp in zip(iterations, improved) if imp]
    improve_costs = [best_cost[i] for i in improve_iters]
    
    if improve_iters:
        ax.scatter(improve_iters, improve_costs, c=COLORS['success'], 
                  s=50, zorder=5, label='改进点', marker='v')
    
    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('目标函数值', fontsize=12)
    ax.set_title(f'ALNS收敛轨迹{title_suffix}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'fig4a_convergence'))
    plt.close()


def plot_operator_weights(iteration_history, output_dir):
    """
    Figure 4B: 算子权重演化图（折线图，使用对数坐标以显示所有权重变化）
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    iterations = iteration_history['iterations']
    destroy_weights = np.array(iteration_history['destroy_weights'])
    repair_weights = np.array(iteration_history['repair_weights'])
    
    # === Destroy 算子 ===
    ax1 = axes[0]
    colors_d = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
    for i, name in enumerate(DESTROY_OP_NAMES):
        ax1.plot(iterations, destroy_weights[:, i], label=name, 
                color=colors_d[i], linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('迭代次数', fontsize=11)
    ax1.set_ylabel('权重 (对数坐标)', fontsize=11)
    ax1.set_title('(A) Destroy 算子权重演化 (N=10)', fontsize=12, fontweight='bold')
    ax1.set_yscale('log')  # 使用对数坐标
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3, which='both')
    
    # === Repair 算子 ===
    ax2 = axes[1]
    colors_r = [COLORS['success'], COLORS['warning'], COLORS['danger']]
    for i, name in enumerate(REPAIR_OP_NAMES):
        ax2.plot(iterations, repair_weights[:, i], label=name,
                color=colors_r[i], linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('迭代次数', fontsize=11)
    ax2.set_ylabel('权重 (对数坐标)', fontsize=11)
    ax2.set_title('(B) Repair 算子权重演化 (N=10)', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')  # 使用对数坐标
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'fig4b_operator_weights'))
    plt.close()


def plot_sa_acceptance(iteration_history, output_dir):
    """
    Figure 5: 模拟退火接受准则可视化
    展示温度下降曲线和接受率变化
    """
    fig, ax1 = plt.subplots(figsize=(12, 5))
    
    iterations = iteration_history['iterations']
    temperature = iteration_history['temperature']
    accepted = iteration_history['accepted']
    improved = iteration_history.get('improved', [False] * len(accepted))
    current_cost = iteration_history.get('current_cost', [])
    
    # 计算每 20 次迭代的统计数据
    window_size = 20
    windows = []
    acceptance_rates = []
    avg_temps = []
    has_change = []  # 该窗口内是否有 cost 变化
    
    for i in range(0, len(iterations), window_size):
        end = min(i + window_size, len(iterations))
        stage_accepted = accepted[i:end]
        stage_improved = improved[i:end]
        stage_temps = temperature[i:end]
        stage_costs = current_cost[i:end] if current_cost else []
        
        n_total = len(stage_accepted)
        n_accepted = sum(1 for a in stage_accepted if a)
        n_improved = sum(1 for im in stage_improved if im)
        
        # 检查是否有 cost 变化
        cost_changed = False
        if len(stage_costs) > 1:
            cost_changed = any(stage_costs[j] != stage_costs[j+1] for j in range(len(stage_costs)-1))
        
        n_worse_accepted = n_accepted - n_improved
        n_not_improved = n_total - n_improved
        
        if n_not_improved > 0 and cost_changed:
            rate = n_worse_accepted / n_not_improved * 100
        else:
            rate = None  # 标记为无意义（收敛后）
        
        windows.append(i + window_size // 2)
        acceptance_rates.append(rate)
        avg_temps.append(np.mean(stage_temps))
        has_change.append(cost_changed)
    
    # 温度曲线（完整）
    color1 = COLORS['primary']
    ax1.semilogy(iterations, temperature, '-', color=color1, 
                linewidth=2, alpha=0.8, label='温度')
    ax1.set_xlabel('迭代次数', fontsize=12)
    ax1.set_ylabel('温度 (对数坐标)', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # 接受率曲线（右轴）- 只绘制有意义的点
    ax2 = ax1.twinx()
    color2 = COLORS['secondary']
    
    # 分离有意义和无意义的数据
    # 直接所有点都画，None 的点设为 0
    valid_windows = windows
    valid_rates = [r if r is not None else 0 for r in acceptance_rates]

    # 自动截断：只显示到最后一次非零接受率后的一段，避免后面全是0太长
    last_non_zero_idx = 0
    for i in range(len(valid_rates)):
        if valid_rates[i] > 0:
            last_non_zero_idx = i
            
    # 多保留 5 个点作为缓冲，展示收敛趋势
    display_end_idx = min(len(valid_windows), last_non_zero_idx + 6)
    
    display_windows = valid_windows[:display_end_idx]
    display_rates = valid_rates[:display_end_idx]

    if display_windows:
        ax2.plot(display_windows, display_rates, 's-', color=color2, 
                linewidth=2, markersize=6, label='劣解接受率')
        
        # 调整 X 轴范围，聚焦于有效区域
        max_iter = display_windows[-1] if display_windows else iterations[-1]
        ax1.set_xlim(left=0, right=max_iter * 1.05) # 留一点右边距
    else:
        # 最后的兜底
        ax2.plot(iterations, [0]*len(iterations), 's-', color=color2,
                 alpha=0.0, label='劣解接受率(无)')

    # 标注收敛点 (First active convergence point)
    # 如果被截断了，收敛点可能就在图的最右侧
    if last_non_zero_idx < len(valid_windows) - 1:
        conv_x = valid_windows[last_non_zero_idx]
        ax2.annotate('开始收敛 (接受率→0)', 
                    xy=(conv_x, 0), xytext=(conv_x, 20),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    fontsize=10, ha='center')
    
    ax2.set_ylabel('劣解接受率 (%)', fontsize=12, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 100)
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax1.set_title('模拟退火温度与劣解接受率演化 (N=10)', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'fig5_sa_acceptance'))
    plt.close()


def plot_time_space_diagram(detailed_log, output_dir, filename_suffix=''):
    """
    Figure 6/7: 时空轨迹图
    X轴：时间，Y轴：楼层
    不同颜色代表不同机器人
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 解析日志，构建机器人轨迹
    robot_trajectories = defaultdict(list)
    robot_colors = {}
    color_palette = plt.cm.Set1.colors
    
    for event in detailed_log:
        time = event['time']
        event_type = event['event']
        data = event.get('data', {})
        
        if event_type == 'ROBOT_MOVE':
            robot_id = data.get('robot_id')
            if robot_id is not None:
                robot_trajectories[robot_id].append({
                    'time': time,
                    'floor': 0,  # 从大厅出发
                    'type': 'start'
                })
                if robot_id not in robot_colors:
                    robot_colors[robot_id] = color_palette[robot_id % len(color_palette)]
        
        elif event_type == 'ROBOT_WAIT':
            robot_id = data.get('robot_id')
            if robot_id is not None:
                robot_trajectories[robot_id].append({
                    'time': time,
                    'floor': 0,
                    'type': 'wait'
                })
        
        elif event_type == 'ELEVATOR_ARRIVE':
            floor = data.get('floor', 0)
            passengers = data.get('passengers', [])
            for robot_id in passengers:
                robot_trajectories[robot_id].append({
                    'time': time,
                    'floor': floor,
                    'type': 'arrive'
                })
        
        elif event_type == 'ROBOT_DELIVER':
            robot_id = data.get('robot_id')
            floor = data.get('floor', 0)
            if robot_id is not None:
                robot_trajectories[robot_id].append({
                    'time': time,
                    'floor': floor,
                    'type': 'deliver'
                })
        
        elif event_type == 'TASK_COMPLETE':
            robot_id = data.get('robot_id')
            if robot_id is not None:
                completion_time = data.get('completion_time', time)
                robot_trajectories[robot_id].append({
                    'time': completion_time,
                    'floor': robot_trajectories[robot_id][-1]['floor'] if robot_trajectories[robot_id] else 0,
                    'type': 'complete'
                })
        
        # ROBOT_RETURN 事件不绘制，因为返程没有详细的电梯到达数据
        # 轨迹在任务完成点自然结束
    
    # 绘制轨迹
    legend_handles = []
    for robot_id, trajectory in sorted(robot_trajectories.items()):
        if not trajectory:
            continue
        
        color = robot_colors.get(robot_id, 'gray')
        
        # 排序并绘制
        trajectory = sorted(trajectory, key=lambda x: x['time'])
        times = [t['time'] for t in trajectory]
        floors = [t['floor'] for t in trajectory]
        
        line, = ax.plot(times, floors, 'o-', color=color, linewidth=2, 
                       markersize=4, alpha=0.8)
        legend_handles.append(Line2D([0], [0], color=color, linewidth=2, 
                                     label=f'Robot {robot_id}'))
        
        # 标记关键事件
        for t in trajectory:
            if t['type'] == 'deliver':
                ax.scatter(t['time'], t['floor'], c=color, s=100, 
                          marker='s', zorder=5, edgecolors='white')
            elif t['type'] == 'complete':
                ax.scatter(t['time'], t['floor'], c='green', s=80, 
                          marker='*', zorder=5)
    
    ax.set_xlabel('时间 (秒)', fontsize=12)
    ax.set_ylabel('楼层', fontsize=12)
    ax.set_title(f'机器人时空轨迹图{filename_suffix}', fontsize=14, fontweight='bold')
    ax.legend(handles=legend_handles, loc='upper right', ncol=2)
    ax.grid(alpha=0.3)
    ax.set_ylim(-5, 120)
    
    # 添加楼层参考线
    ax.axhline(y=0, color=COLORS['gray'], linestyle='--', alpha=0.5, label='大厅层')
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, f'fig6_trajectory{filename_suffix}'))
    plt.close()


def plot_elevator_heatmap(detailed_log, output_dir):
    """
    Figure 8: 电梯资源利用热力图
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # 找出最大时间
    max_time = max(event['time'] for event in detailed_log) if detailed_log else 500
    max_time = min(max_time + 50, 800)  # 限制最大时间
    
    # 更精细的时间分辨率
    time_bins = np.arange(0, max_time, 10)  # 10秒一个时间段
    num_elevators = 2
    utilization = np.zeros((num_elevators, len(time_bins) - 1))
    
    # 统计多种电梯相关事件
    for event in detailed_log:
        event_type = event['event']
        time = event['time']
        data = event.get('data', {})
        
        # 统计 ELEVATOR_LOAD 事件
        if event_type == 'ELEVATOR_LOAD':
            elevator_id = data.get('elevator_id', 0)
            robots = data.get('robots', [])
            
            bin_idx = np.searchsorted(time_bins, time) - 1
            if 0 <= bin_idx < len(time_bins) - 1 and elevator_id < num_elevators:
                utilization[elevator_id, bin_idx] += len(robots)
        
        # 统计 ROBOT_WAIT 事件（机器人等待电梯）
        elif event_type == 'ROBOT_WAIT':
            elevator_id = data.get('elevator_id', 0)
            
            bin_idx = np.searchsorted(time_bins, time) - 1
            if 0 <= bin_idx < len(time_bins) - 1 and elevator_id < num_elevators:
                utilization[elevator_id, bin_idx] += 0.5  # 等待也算占用
        
        # 统计 ELEVATOR_ARRIVE 事件
        elif event_type == 'ELEVATOR_ARRIVE':
            elevator_id = data.get('elevator_id', 0)
            passengers = data.get('passengers', [])
            
            bin_idx = np.searchsorted(time_bins, time) - 1
            if 0 <= bin_idx < len(time_bins) - 1 and elevator_id < num_elevators:
                utilization[elevator_id, bin_idx] += len(passengers) * 0.3
    
    # 绘制热力图
    im = ax.imshow(utilization, cmap='YlOrRd', aspect='auto', 
                  extent=[time_bins[0], time_bins[-1], -0.5, num_elevators - 0.5],
                  vmin=0, vmax=max(4, np.max(utilization)))
    
    ax.set_yticks(range(num_elevators))
    ax.set_yticklabels([f'电梯 E{i}' for i in range(num_elevators)])
    ax.set_xlabel('时间 (秒)', fontsize=12)
    ax.set_ylabel('电梯', fontsize=12)
    ax.set_title('电梯资源利用热力图', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('利用强度', fontsize=11)
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'fig8_elevator_heatmap'))
    plt.close()


def plot_elevator_heatmap_with_scale(detailed_log, output_dir, scale_label):
    """
    Figure 8: 电梯资源利用热力图（带规模标注）
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    
    max_time = max(event['time'] for event in detailed_log) if detailed_log else 500
    max_time = min(max_time + 50, 800)
    
    time_bins = np.arange(0, max_time, 10)
    num_elevators = 2
    utilization = np.zeros((num_elevators, len(time_bins) - 1))
    
    for event in detailed_log:
        event_type = event['event']
        time = event['time']
        data = event.get('data', {})
        
        if event_type == 'ELEVATOR_LOAD':
            elevator_id = data.get('elevator_id', 0)
            robots = data.get('robots', [])
            bin_idx = np.searchsorted(time_bins, time) - 1
            if 0 <= bin_idx < len(time_bins) - 1 and elevator_id < num_elevators:
                utilization[elevator_id, bin_idx] += len(robots)
        
        elif event_type == 'ROBOT_WAIT':
            elevator_id = data.get('elevator_id', 0)
            bin_idx = np.searchsorted(time_bins, time) - 1
            if 0 <= bin_idx < len(time_bins) - 1 and elevator_id < num_elevators:
                utilization[elevator_id, bin_idx] += 0.5
        
        elif event_type == 'ELEVATOR_ARRIVE':
            elevator_id = data.get('elevator_id', 0)
            passengers = data.get('passengers', [])
            bin_idx = np.searchsorted(time_bins, time) - 1
            if 0 <= bin_idx < len(time_bins) - 1 and elevator_id < num_elevators:
                utilization[elevator_id, bin_idx] += len(passengers) * 0.3
    
    im = ax.imshow(utilization, cmap='YlOrRd', aspect='auto', 
                  extent=[time_bins[0], time_bins[-1], -0.5, num_elevators - 0.5],
                  vmin=0, vmax=max(4, np.max(utilization)))
    
    ax.set_yticks(range(num_elevators))
    ax.set_yticklabels([f'电梯 E{i}' for i in range(num_elevators)])
    ax.set_xlabel('时间 (秒)', fontsize=12)
    ax.set_ylabel('电梯', fontsize=12)
    ax.set_title(f'电梯资源利用热力图 ({scale_label})', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('利用强度', fontsize=11)
    
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'fig8_elevator_heatmap'))
    plt.close()


def plot_metric_comparison_small(results_by_algo, output_dir, target_n=6):
    """
    Figure 2B: 小规模多指标对比图（三算法，仅N=6）
    """
    metrics = ['objective_value', 'makespan', 'on_time_rate', 'run_time']
    metric_labels = ['目标函数值', '完工时间(s)', '准时率(%)', '运行时间(s)']
    metric_scales = [1, 1, 100, 1]
    
    algo_names = list(results_by_algo.keys())
    algo_values = {algo: [] for algo in algo_names}
    
    for algo, results in results_by_algo.items():
        # 只使用指定规模的数据
        filtered_results = [r for r in results if r.get('num_tasks') == target_n]
        if not filtered_results:
            filtered_results = results  # 如果没有指定规模数据，使用全部
        
        for i, metric in enumerate(metrics):
            values = [r.get(metric, 0) * metric_scales[i] for r in filtered_results if metric in r]
            if values:
                algo_values[algo].append(np.mean(values))
            else:
                algo_values[algo].append(0)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    x = np.arange(len(algo_names))
    width = 0.6
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        values = [algo_values[algo][i] for algo in algo_names]
        colors = [COLORS.get(algo, COLORS['gray']) for algo in algo_names]
        
        bars = ax.bar(x, values, width, color=colors, alpha=0.8, edgecolor='white')
        
        ax.set_ylabel(label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(algo_names, fontsize=10)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if metric == 'on_time_rate':
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
            elif metric == 'run_time':
                # 统一用秒显示
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}s', ha='center', va='bottom', fontsize=9)
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'多指标性能对比 (N={target_n})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, 'fig2b_metric_comparison_small'))
    plt.close()

def plot_radar_chart(results_by_algo, output_dir, title_suffix=""):
    """
    Figure 2: 多指标对比图（分组柱状图，更直观）
    """
    # 指标配置
    metrics = ['objective_value', 'makespan', 'on_time_rate', 'run_time']
    metric_labels = ['目标函数值', '完工时间(s)', '准时率(%)', '运行时间(s)']
    metric_scales = [1, 1, 100, 1]  # 准时率需要乘100
    
    # 计算每个算法的平均值
    algo_names = list(results_by_algo.keys())
    algo_values = {algo: [] for algo in algo_names}
    
    for algo, results in results_by_algo.items():
        for i, metric in enumerate(metrics):
            values = [r.get(metric, 0) * metric_scales[i] for r in results if metric in r]
            if values:
                algo_values[algo].append(np.mean(values))
            else:
                algo_values[algo].append(0)
    
    # 创建 2x2 子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    x = np.arange(len(algo_names))
    width = 0.6
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        values = [algo_values[algo][i] for algo in algo_names]
        colors = [COLORS.get(algo, COLORS['gray']) for algo in algo_names]
        
        bars = ax.bar(x, values, width, color=colors, alpha=0.8, edgecolor='white')
        
        ax.set_ylabel(label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(algo_names, fontsize=10)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 在柱子上显示数值
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if metric == 'on_time_rate':
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
            elif metric == 'run_time':
                # 统一用秒显示
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}s', ha='center', va='bottom', fontsize=9)
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    
    if title_suffix:
        plt.suptitle(f'算法多指标对比 {title_suffix}', fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    # 留出标题空间
    if title_suffix:
        plt.subplots_adjust(top=0.9)
        
    save_figure(fig, os.path.join(output_dir, 'fig2_metric_comparison'))
    plt.close()


def generate_table_data(iteration_history, initial_assignment=None):
    """
    生成论文表格所需的真实数据
    返回用于替换论文中虚假数据的真实数据
    """
    tables = {}
    
    # 表6: 算子权重演化过程
    sample_iters = [0, 20, 40, 60, 80, 100]
    table6_data = []
    
    for it in sample_iters:
        if it < len(iteration_history['iterations']):
            idx = it
            row = {
                '迭代次数': it,
                'Random': round(iteration_history['destroy_weights'][idx][0], 2),
                'Worst': round(iteration_history['destroy_weights'][idx][1], 2),
                'Related': round(iteration_history['destroy_weights'][idx][2], 2),
                'Smart': round(iteration_history['repair_weights'][idx][0], 2),
                'Regret-2': round(iteration_history['repair_weights'][idx][1], 2),
                'Batch': round(iteration_history['repair_weights'][idx][2], 2),
            }
            table6_data.append(row)
    
    tables['table6_operator_weights'] = table6_data
    
    # 表7: 解接受情况统计
    stage_size = 200
    table7_data = []
    
    for i in range(0, min(1000, len(iteration_history['iterations'])), stage_size):
        end = min(i + stage_size, len(iteration_history['iterations']))
        stage = f'{i}-{end}'
        
        temps = iteration_history['temperature'][i:end]
        accepted = iteration_history['accepted'][i:end]
        improved = iteration_history['improved'][i:end]
        current_costs = iteration_history['current_cost'][i:end]
        best_costs = iteration_history['best_cost'][i:end]
        
        # 统计
        improve_count = sum(improved)
        
        # 劣解统计（当前解比最优解差的情况）
        worse_mask = [c > b for c, b in zip(current_costs, best_costs)]
        worse_count = sum(worse_mask)
        accepted_worse = sum(a for a, w in zip(accepted, worse_mask) if w)
        rejected_worse = worse_count - accepted_worse
        
        if worse_count > 0:
            worse_accept_rate = round(accepted_worse / worse_count * 100, 1)
        else:
            worse_accept_rate = 0
        
        row = {
            '迭代阶段': stage,
            '温度': f'{temps[0]:.1f}-{temps[-1]:.1f}',
            '改进解': improve_count,
            '接受劣解': accepted_worse,
            '拒绝劣解': rejected_worse,
            '劣解接受率(%)': worse_accept_rate,
        }
        table7_data.append(row)
    
    tables['table7_acceptance_stats'] = table7_data
    
    return tables


def print_table_markdown(table_data, table_name):
    """以 Markdown 格式打印表格"""
    if not table_data:
        return
    
    print(f"\n### {table_name}\n")
    
    headers = list(table_data[0].keys())
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(['---'] * len(headers)) + " |")
    
    for row in table_data:
        values = [str(row[h]) for h in headers]
        print("| " + " | ".join(values) + " |")
