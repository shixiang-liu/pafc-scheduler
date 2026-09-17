"""数据导出工具"""

import json
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from core.config import SystemConfig


def export_to_json(results, filepath, metadata=None):
    """导出JSON"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    export_data = {
        'metadata': metadata or {},
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_robots': SystemConfig.NUM_ROBOTS,
            'num_elevators': SystemConfig.NUM_ELEVATORS,
            'elevator_capacity': SystemConfig.ELEVATOR_CAPACITY,
        },
        'results': results,
    }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    return str(path)


def export_to_csv(results, filepath):
    """导出CSV"""
    if not results:
        return ""

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    priority_keys = ['algorithm', 'seed', 'num_robots', 'num_tasks',
                     'objective_value', 'on_time_rate', 'run_time']

    all_keys = set()
    for record in results:
        all_keys.update(record.keys())

    other_keys = sorted(all_keys - set(priority_keys))
    fieldnames = [k for k in priority_keys if k in all_keys] + other_keys

    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return str(path)


def generate_latex_table(results, caption="算法性能对比", label="tab:comparison"):
    """生成LaTeX表格"""
    if not results:
        return ""

    stats = defaultdict(list)

    for r in results:
        if 'algorithm' not in r:
            continue
        algo = r['algorithm']
        stats[algo].append(r)

    algorithms = sorted(stats.keys())

    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\toprule")
    latex.append("算法 & 目标函数值 & 准时率(\\%) & 运行时间(秒) & 样本数 \\\\")
    latex.append("\\midrule")

    for algo in algorithms:
        records = stats[algo]
        n = len(records)

        obj_vals = [r.get('objective_value', 0) for r in records]
        ontime_vals = [r.get('on_time_rate', 0) * 100 for r in records]
        time_vals = [r.get('run_time', 0) for r in records]

        obj_mean = sum(obj_vals) / n if n > 0 else 0
        obj_std = (sum((v - obj_mean)**2 for v in obj_vals) / n) ** 0.5 if n > 1 else 0

        ontime_mean = sum(ontime_vals) / n if n > 0 else 0
        ontime_std = (sum((v - ontime_mean)**2 for v in ontime_vals) / n) ** 0.5 if n > 1 else 0

        time_mean = sum(time_vals) / n if n > 0 else 0
        time_std = (sum((v - time_mean)**2 for v in time_vals) / n) ** 0.5 if n > 1 else 0

        latex.append(
            f"{algo} & "
            f"{obj_mean:.1f}$\\pm${obj_std:.1f} & "
            f"{ontime_mean:.1f}$\\pm${ontime_std:.1f} & "
            f"{time_mean:.3f}$\\pm${time_std:.3f} & "
            f"{n} \\\\"
        )

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def save_latex_table(results, filepath, caption="算法性能对比", label="tab:comparison"):
    """保存LaTeX表格"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    latex_code = generate_latex_table(results, caption, label)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(latex_code)

    return str(path)