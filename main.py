"""平安金融中心垂直物流调度系统 - 主程序入口

三阶段对比实验：
Phase 1: 小规模最优性验证（N=4,6）
Phase 2: 规模扩展性测试（N=10-50）
Phase 3: 大规模工程评估（N=80）
"""

import sys
import time
import json
import os
from datetime import datetime
from collections import defaultdict

from core.config import SystemConfig as Config
from core.task_generator import generate_tasks
from algorithms.greedy import run_greedy
from algorithms.branch_bound import run_bnb
from algorithms.alns import run_alns


def save_results(results, phase_name, output_dir="outputs/results"):
    # 保存实验结果到JSON文件，iteration_history单独存储
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 分离iteration_history和detailed_log以减小主文件体积
    history_data = []
    main_results = []
    
    for result in results:
        # 复制结果，避免修改原始数据
        r = dict(result)
        
        # 提取iteration_history
        if 'iteration_history' in r:
            history_data.append({
                'algorithm': r.get('algorithm'),
                'seed': r.get('seed'),
                'num_tasks': r.get('num_tasks'),
                'iteration_history': r.pop('iteration_history')
            })
        
        # 移除detailed_log
        if 'detailed_log' in r:
            del r['detailed_log']
        
        main_results.append(r)
    
    # 保存主结果
    filename = f"{phase_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(main_results, f, indent=2, ensure_ascii=False)
    print(f"✓ 结果已保存: {filepath}")
    
    # 保存iteration_history到单独文件
    if history_data:
        history_file = f"{phase_name}_history_{timestamp}.json"
        history_path = os.path.join(output_dir, history_file)
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        print(f"✓ 迭代历史已保存: {history_path}")
    
    return filepath


def print_summary(results):
    """打印实验结果摘要"""
    if not results:
        return

    # 按算法分组统计
    groups = defaultdict(list)
    for r in results:
        groups[r['algorithm']].append(r)

    print("\n" + "=" * 120)
    print("实验结果摘要")
    print("=" * 120)

    # 算法顺序
    algo_order = ['B&B', 'ALNS', 'Greedy-EDF']
    available_algos = [a for a in algo_order if a in groups]

    # 表头
    header = f"{'算法':^14} | {'目标值':^16} | {'准时率':^8} | {'拼梯':^7} | {'多任务':^6} | {'死锁':^7} | {'运行时间':^10}"
    print(header)
    print("-" * 120)

    best_obj = float('inf')
    best_algo = ""

    for algo in available_algos:
        records = groups[algo]
        n = len(records)

        # 计算各指标的均值和标准差
        obj_vals = [r['objective_value'] for r in records]
        ontime_vals = [r['on_time_rate'] * 100 for r in records]
        batch_vals = [r.get('batching_count', 0) for r in records]
        multi_vals = [r.get('multi_task_count', 0) for r in records]
        deadlock_vals = [r.get('deadlock_count', 0) for r in records]
        time_vals = [r['run_time'] for r in records]

        obj_mean = sum(obj_vals) / n
        obj_std = (sum((v - obj_mean) ** 2 for v in obj_vals) / n) ** 0.5 if n > 1 else 0
        ontime_mean = sum(ontime_vals) / n
        batch_mean = sum(batch_vals) / n
        multi_mean = sum(multi_vals) / n
        deadlock_mean = sum(deadlock_vals) / n
        time_mean = sum(time_vals) / n

        if obj_mean < best_obj:
            best_obj = obj_mean
            best_algo = algo

        # 输出行
        obj_str = f"{obj_mean:.1f}±{obj_std:.1f}"
        ontime_str = f"{ontime_mean:.1f}%"
        batch_str = f"{batch_mean:.1f}"
        multi_str = f"{multi_mean:.1f}"
        deadlock_str = f"{deadlock_mean:.1f}"
        time_str = f"{time_mean:.3f}s"

        print(f"{algo:^15} | {obj_str:^18} | {ontime_str:^10} | {batch_str:^8} | "
              f"{multi_str:^8} | {deadlock_str:^8} | {time_str:^12}")

    print("-" * 120)
    print(f"✓ 最优算法: {best_algo} (目标值={best_obj:.1f})")

    # 计算相对基准算法的改进率
    if 'Greedy-EDF' in groups:
        greedy_obj = sum(r['objective_value'] for r in groups['Greedy-EDF']) / len(groups['Greedy-EDF'])
        if greedy_obj > 0:
            print(f"\n相对 Greedy-EDF 改进率:")
            for algo in available_algos:
                if algo != 'Greedy-EDF':
                    algo_obj = sum(r['objective_value'] for r in groups[algo]) / len(groups[algo])
                    improvement = (greedy_obj - algo_obj) / greedy_obj * 100
                    print(f"  {algo:12s}: {improvement:+.2f}%")

    print("=" * 120)


def run_experiment(num_robots, num_tasks, seeds, algorithms, enable_log=False):
    """运行一组实验"""
    all_results = []
    Config.NUM_ROBOTS = num_robots

    print(f"\n✓ 配置: {num_robots}机器人 × {num_tasks}任务 × {len(seeds)}组测试用例")

    for i, seed in enumerate(seeds, 1):
        print(f"\r  进度: [{i}/{len(seeds)}] ", end='', flush=True)

        tasks = generate_tasks(num_tasks, seed)

        for algo_name, algo_func in algorithms:
            result = algo_func(tasks, seed=seed, enable_log=enable_log)
            result.update({
                "algorithm": algo_name,
                "seed": seed,
                "num_robots": num_robots,
                "num_tasks": num_tasks
            })
            all_results.append(result)

    print()
    return all_results


def phase1_optimality():
    """阶段1: 小规模最优性验证（N=4,6）"""
    print(f"\n{'='*80}")
    print("  Phase 1: 小规模最优性验证（N=4,6）")
    print("  说明：对比B&B（精确）vs ALNS（启发式）vs Greedy（基准）")
    print(f"{'='*80}")

    phase1_results = []

    task_sizes = [4, 6]
    Config.NUM_ELEVATORS = 2
    Config.ELEVATOR_CAPACITY = 4
    seeds = [11, 21, 31, 41, 51]
    all_results = []

    for task_idx, num_tasks in enumerate(task_sizes):
        print(f"\n【任务规模 N={num_tasks}】")
        print("-" * 80)

        Config.NUM_ROBOTS = 4

        for i, seed in enumerate(seeds, 1):
            # 只在第一个规模(N=4)的第一个种子启用详细日志并打印
            is_first_case = (task_idx == 0 and i == 1)
            enable_log = is_first_case

            print(f"\r  进度: [{i}/{len(seeds)}] ", end='', flush=True)
            tasks = generate_tasks(num_tasks, seed)

            # B&B设置合理的时间限制
            algorithms = [
                ("B&B", lambda tasks, seed, enable_log:
                 run_bnb(tasks, max_nodes=500000, time_limit=120.0, seed=seed, enable_log=enable_log)),
                ("ALNS", lambda tasks, seed, enable_log:
                 run_alns(tasks, max_iterations=500, seed=seed, enable_log=enable_log)),
                ("Greedy-EDF", lambda tasks, seed, enable_log:
                 run_greedy(tasks, seed=seed, enable_log=enable_log)),
            ]

            for algo_name, algo_func in algorithms:
                result = algo_func(tasks, seed=seed, enable_log=enable_log)
                result.update({
                    "algorithm": algo_name,
                    "seed": seed,
                    "num_robots": 4,
                    "num_tasks": num_tasks
                })
                all_results.append(result)

                # 只在N=6的第一个种子、只打印B&B算法的详细日志
                if is_first_case and algo_name == "B&B" and result.get('detailed_log'):
                    print(f"\n\n{'='*80}")
                    print(f"  仿真过程详细日志 - {algo_name} (N={num_tasks}, seed={seed})")
                    print(f"{'='*80}")
                    for log in result['detailed_log']:
                        event = log['event']
                        msg = log['message']
                        t = log['time']
                        print(f"  [{t:7.1f}s] {event:16s} | {msg}")
                    print(f"{'='*80}\n")

        print()
        # 过滤当前规模的结果
        current_results = [r for r in all_results if r['num_tasks'] == num_tasks]
        print_summary(current_results)

    save_results(all_results, "phase1")
    return all_results


def phase2_scalability():
    """阶段2: 规模扩展性测试（N=8-50）"""
    print("\n" + "=" * 80)
    print("  Phase 2: 规模扩展性测试（N=8-50）")
    print("  说明：测试ALNS在不同规模下的表现")
    print("=" * 80)

    Config.NUM_ELEVATORS = 2
    Config.ELEVATOR_CAPACITY = 4

    # 从N=10开始
    scale_configs = [(5, 10), (10, 20), (15, 30), (20, 40), (25, 50)]
    seeds = [201, 202, 203, 204, 205]
    all_results = []

    for num_robots, num_tasks in scale_configs:
        print(f"\n【任务规模 N={num_tasks}, 机器人={num_robots}】")
        print("-" * 80)

        algorithms = [
            ("ALNS", lambda tasks, seed, enable_log:
             run_alns(tasks, max_iterations=500, seed=seed, enable_log=enable_log)),
            ("Greedy-EDF", lambda tasks, seed, enable_log:
             run_greedy(tasks, seed=seed, enable_log=enable_log)),
        ]

        results = run_experiment(num_robots, num_tasks, seeds, algorithms)
        all_results.extend(results)
        print_summary(results)

    save_results(all_results, "phase2")
    return all_results


def phase3_engineering():
    """阶段3: 大规模工程效能评估（N=80）"""
    print("\n" + "=" * 80)
    print("  Phase 3: 大规模工程效能评估（N=80）")
    print("  说明：验证ALNS在实际规模下的工程可行性")
    print("=" * 80)

    Config.NUM_ELEVATORS = 2
    Config.ELEVATOR_CAPACITY = 4
    num_robots = 40
    num_tasks = 80
    seeds = [301, 302, 303, 304, 305]

    print(f"\n【大规模场景: {num_robots}机器人 × {num_tasks}任务】")
    print("-" * 80)

    algorithms = [
        ("ALNS", lambda tasks, seed, enable_log:
         run_alns(tasks, max_iterations=500, seed=seed, enable_log=enable_log)),
        ("Greedy-EDF", lambda tasks, seed, enable_log:
         run_greedy(tasks, seed=seed, enable_log=enable_log)),
    ]

    results = run_experiment(num_robots, num_tasks, seeds, algorithms)
    print_summary(results)

    save_results(results, "phase3")
    return results


def main():
    """主程序入口"""
    print("\n" + "=" * 80)
    print("  平安金融中心垂直物流调度系统 - 三阶段对比实验")
    print("  算法对比: B&B vs ALNS vs Greedy-EDF")
    print("=" * 80)
    print("\n系统配置:")
    print(f"  电梯: {Config.NUM_ELEVATORS}部 × 容量{Config.ELEVATOR_CAPACITY} = "
          f"{Config.NUM_ELEVATORS * Config.ELEVATOR_CAPACITY}位")
    print(f"  任务deadline: {Config.DEFAULT_DEADLINE}秒 ({Config.DEFAULT_DEADLINE / 60:.1f}分钟)")
    print(f"  目标权重: 时间={Config.WEIGHT_TIME}, 能耗={Config.WEIGHT_ENERGY}, "
          f"超时={Config.WEIGHT_PENALTY}")

    print("\n实验说明:")
    print("  Phase 1: N=4,6 - 对比B&B（精确）vs ALNS（启发式）vs Greedy（基准）")
    print("  Phase 2: N=10-50 - 测试ALNS的规模扩展性")
    print("  Phase 3: N=80 - 验证ALNS在实际场景的工程可行性")

    phases = {
        "1": ("阶段1: 小规模最优性验证", phase1_optimality),
        "2": ("阶段2: 规模扩展性测试", phase2_scalability),
        "3": ("阶段3: 大规模工程效能评估", phase3_engineering),
        "all": ("全部阶段", lambda: [phase1_optimality(), phase2_scalability(), phase3_engineering()]),
    }

    # 命令行模式
    if len(sys.argv) > 1:
        choice = sys.argv[1].lower()
        if choice not in phases:
            print(f"\n✗ 错误: 未知选项 '{choice}'")
            print("可用选项: 1 | 2 | 3 | all")
            sys.exit(1)

        start_time = time.time()
        phases[choice][1]()
        print(f"\n✓ 总耗时: {time.time() - start_time:.1f}秒")
        return

    # 交互模式
    print("\n请选择要运行的实验阶段:")
    for key, (desc, _) in phases.items():
        print(f"  {key} - {desc}")

    choice = input("\n请输入选项 (1/2/3/all): ").strip().lower()
    if choice not in phases:
        print(f"✗ 错误: 无效选项 '{choice}'")
        sys.exit(1)

    start_time = time.time()
    phases[choice][1]()
    print(f"\n✓ 总耗时: {time.time() - start_time:.1f}秒")


if __name__ == "__main__":
    main()