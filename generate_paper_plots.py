"""论文图表生成主脚本

运行此脚本生成所有论文所需的图表和表格数据。
输出目录: outputs/paper/
"""

import os
import sys
import json
import random
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg') # 解决 Windows 下 Qt 插件错误

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.task_generator import generate_tasks
from core.environment import SimulationEnvironment
from algorithms.alns import run_alns
from algorithms.greedy import run_greedy
from utils.paper_plots import (
    plot_performance_overview,
    plot_metric_comparison_small, # 添加此函数
    plot_scalability,
    plot_convergence,
    plot_operator_weights,
    plot_sa_acceptance,
    plot_time_space_diagram,
    plot_elevator_heatmap,
    plot_radar_chart,
    generate_table_data,
    save_figure,
    COLORS
)


def load_results(phase_name):
    """加载实验结果（排除历史文件）"""
    results_dir = 'outputs/results'
    
    # 查找最新的结果文件（排除_history文件）
    files = [f for f in os.listdir(results_dir) 
             if f.startswith(phase_name) and f.endswith('.json') and '_history' not in f]
    if not files:
        print(f"[!] 未找到 {phase_name} 的结果文件")
        return []
    
    latest_file = sorted(files)[-1]
    filepath = os.path.join(results_dir, latest_file)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"[✓] 加载 {phase_name}: {latest_file} ({len(data)} 条记录)")
    return data


def run_alns_with_history(num_tasks, seed, enable_log=True):
    """运行 ALNS 并返回带有迭代历史的结果"""
    print(f"\n[ALNS] 运行实验: N={num_tasks}, seed={seed}")
    
    tasks = generate_tasks(num_tasks, seed)
    result = run_alns(tasks, seed=seed, enable_log=enable_log)
    
    return result


def main():
    """主函数：生成所有论文图表"""
    
    # 创建输出目录
    output_dir = 'outputs/paper'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("论文图表生成器")
    print("=" * 60)
    
    # 加载现有实验结果
    print("\n[1/6] 加载实验结果...")
    phase1_data = load_results('phase1')
    phase2_data = load_results('phase2')
    phase3_data = load_results('phase3')
    
    all_data = phase1_data + phase2_data + phase3_data
    
    # 运行带历史记录的 ALNS 实验
    print("\n[2/6] 运行 ALNS 实验以获取迭代历史...")
    
    # N=10 用于表格数据
    alns_n10_result = run_alns_with_history(10, seed=201, enable_log=True)
    
    # N=6 用于小规模轨迹图 (改用 Greedy 以匹配表2的案例分析)
    print("  运行 Greedy-EDF (N=6) 实验以匹配表2...")
    tasks_n6 = generate_tasks(6, seed=11)
    greedy_n6_result = run_greedy(tasks_n6, seed=11, enable_log=True)
    
    # N=20 用于大规模轨迹图
    alns_n20_result = run_alns_with_history(20, seed=201, enable_log=True)
    
    # 生成图表
    print("\n[3/6] 生成性能对比图表...")
    
    if phase1_data and phase2_data:
        plot_performance_overview(phase1_data, phase2_data, output_dir)
        print("  [✓] Figure 1: 性能全景图")

    # Figure 2B: 小规模多指标对比图 (N=6)
    if phase1_data:
        results_by_algo_p1 = defaultdict(list)
        for r in phase1_data:
            if 'algorithm' in r:
                results_by_algo_p1[r['algorithm']].append(r)
        
        # 确保包含 needed algorithms
        if 'B&B' in results_by_algo_p1:
            plot_metric_comparison_small(results_by_algo_p1, output_dir, target_n=6)
            print("  [✓] Figure 2B: 小规模多指标对比图 (N=6)")
    
    # N=80 多指标对比图 (Figure 2)
    # 优先使用 Phase 3 (N=80) 的数据
    target_data = phase3_data if phase3_data else phase2_data
    source_name = "N=80 (Phase 3)" if phase3_data else "Phase 2"
    
    results_by_algo = {}
    for result in target_data:
        algo = result['algorithm']
        if algo not in results_by_algo:
            results_by_algo[algo] = []
        results_by_algo[algo].append(result)
    
    if results_by_algo:
        # 传递 N=80 标注
        suffix = "(N=80)" if phase3_data else ""
        plot_radar_chart(results_by_algo, output_dir, title_suffix=suffix)
        print(f"  [✓] Figure 2: 多指标对比图 ({source_name})")
    
    # 扩展性图表
    if all_data:
        plot_scalability(all_data, output_dir)
        print("  [✓] Figure 3: 计算效率与扩展性")
    
    print("\n[4/6] 生成 ALNS 算法分析图表...")
    
    # 使用 N=10 来生成 ALNS 收敛图表（N=10有22.9%的改进，更能展示算法动态）
    print("  使用 N=10 ALNS 结果生成收敛图表...")
    
    if 'iteration_history' in alns_n10_result:
        history = alns_n10_result['iteration_history']
        
        plot_convergence(history, output_dir, ' (N=10)')
        print("  [✓] Figure 4A: 收敛轨迹图")
        
        plot_operator_weights(history, output_dir)
        print("  [✓] Figure 4B: 算子权重演化图")
        
    # 为了展示完整的模拟退火接受率变化，专门运行一个参数调整过的 N=10 实验
    # 提高初始温度和减慢冷却速度，强制产生丰富的接受行为
    print("  运行 N=10 (SA演示模式) 实验以生成完整的 SA 曲线...")
    tasks_sa = generate_tasks(10, seed=201)
    
    # 手动创建环境和调度器，以便修改 SA 参数
    from algorithms.alns import ALNSScheduler
    sa_env = SimulationEnvironment()
    sa_env.reset(tasks_sa, seed=201, enable_log=True)
    
    # 使用自定义参数：T=2000 (很高), cooling=0.995 (很慢), max_iterations=2000 (跑久一点)
    # 注意：ALNSScheduler.run() 不接受参数，参数需在 init 或 env 中设置
    sa_scheduler = ALNSScheduler(sa_env, max_iterations=2000, initial_temp=2000, cooling_rate=0.995)
    
    # 运行
    sa_metrics = sa_scheduler.run()
    
    if 'iteration_history' in sa_metrics:
        history = sa_metrics['iteration_history']
        plot_sa_acceptance(history, output_dir)
        print("  [✓] Figure 5: 模拟退火接受准则 (N=10, T=2000, cool=0.995)")
    
    print("\n[5/6] 生成时空轨迹图...")
    
    if 'detailed_log' in greedy_n6_result:
        plot_time_space_diagram(greedy_n6_result['detailed_log'], output_dir, '_n6')
        print("  [✓] Figure 6: 时空轨迹图 (N=6, Greedy)")
    
    if 'detailed_log' in alns_n20_result:
        plot_time_space_diagram(alns_n20_result['detailed_log'], output_dir, '_n20')
        print("  [✓] Figure 7: 时空轨迹图 (N=20)")
        
        # 电梯热力图现在会标注实验规模
        from utils.paper_plots import plot_elevator_heatmap_with_scale
        plot_elevator_heatmap_with_scale(alns_n20_result['detailed_log'], output_dir, 'N=20')
        print("  [✓] Figure 8: 电梯资源利用热力图 (N=20)")
    
    # 生成表格数据
    print("\n[6/6] 生成论文表格数据...")
    
    if 'iteration_history' in alns_n20_result:
        history = alns_n20_result['iteration_history']
        # 只打印 Markdown 格式表格，不保存 JSON（由 generate_tables.py 统一处理）
        print("\n" + "=" * 60)
        print("论文表格数据（Markdown 格式）")
        print("=" * 60)
        
        from utils.paper_plots import generate_table_data, print_table_markdown
        tables = generate_table_data(history)
        print_table_markdown(tables['table6_operator_weights'], 
                            '表6: 算子权重演化过程 (N=20)')
        print_table_markdown(tables['table7_acceptance_stats'], 
                            '表7: 解接受情况统计 (N=20)')
    
    # 汇总
    print("\n" + "=" * 60)
    print("图表生成完成！")
    print("=" * 60)
    print(f"输出目录: {os.path.abspath(output_dir)}")
    
    # 列出生成的文件
    print("\n生成的文件:")
    for f in sorted(os.listdir(output_dir)):
        filepath = os.path.join(output_dir, f)
        size = os.path.getsize(filepath)
        print(f"  - {f} ({size/1024:.1f} KB)")


if __name__ == '__main__':
    main()
