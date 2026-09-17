"""统计分析工具"""

import numpy as np
from scipy import stats


def compute_statistics(results, metric='objective_value'):
    """计算统计量"""
    values = [r[metric] for r in results if metric in r and r[metric] is not None]
    if not values:
        return None

    values = np.array(values)
    n = len(values)

    if n == 0:
        return None

    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0.0
    sem = stats.sem(values) if n > 1 else 0.0

    # 95%置信区间
    if n > 1:
        ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
    else:
        ci = (mean, mean)

    median = np.median(values)
    q25 = np.percentile(values, 25)
    q75 = np.percentile(values, 75)

    return {
        'mean': float(mean),
        'std': float(std),
        'median': float(median),
        'q25': float(q25),
        'q75': float(q75),
        'n': n,
        'ci_lower': float(ci[0]),
        'ci_upper': float(ci[1]),
        'sem': float(sem),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
    }


def compare_algorithms(results_a, results_b,
                      metric='objective_value',
                      algorithm_a_name="算法A",
                      algorithm_b_name="算法B"):
    """比较两个算法"""
    values_a = [r[metric] for r in results_a if metric in r and r[metric] is not None]
    values_b = [r[metric] for r in results_b if metric in r and r[metric] is not None]

    if not values_a or not values_b:
        return None

    values_a = np.array(values_a)
    values_b = np.array(values_b)

    # 配对检验
    min_len = min(len(values_a), len(values_b))
    is_paired = False

    if 'seed' in results_a[0] and 'seed' in results_b[0]:
        seeds_a = {r['seed']: r[metric] for r in results_a if metric in r}
        seeds_b = {r['seed']: r[metric] for r in results_b if metric in r}
        common_seeds = set(seeds_a.keys()) & set(seeds_b.keys())

        if len(common_seeds) >= min_len * 0.8:
            values_a = np.array([seeds_a[s] for s in sorted(common_seeds)])
            values_b = np.array([seeds_b[s] for s in sorted(common_seeds)])
            is_paired = True
            min_len = len(common_seeds)

    if not is_paired and len(values_a) != len(values_b):
        values_a = values_a[:min_len]
        values_b = values_b[:min_len]

    stats_a = compute_statistics(results_a[:min_len], metric)
    stats_b = compute_statistics(results_b[:min_len], metric)

    if not stats_a or not stats_b:
        return None

    # 正态性检验
    is_normal = True
    if min_len <= 50 and min_len >= 3:
        _, p_norm_a = stats.shapiro(values_a)
        _, p_norm_b = stats.shapiro(values_b)
        is_normal = p_norm_a > 0.05 and p_norm_b > 0.05

    # 选择检验方法
    if is_normal and is_paired:
        statistic, p_value = stats.ttest_rel(values_a, values_b)
        test_name = "配对t检验"
    elif is_normal:
        statistic, p_value = stats.ttest_ind(values_a, values_b)
        test_name = "独立样本t检验"
    else:
        statistic, p_value = stats.mannwhitneyu(values_a, values_b, alternative='two-sided')
        test_name = "Mann-Whitney U检验"

    # 效应量
    pooled_std = np.sqrt((stats_a['std']**2 + stats_b['std']**2) / 2)
    if pooled_std > 0:
        cohens_d = (stats_a['mean'] - stats_b['mean']) / pooled_std
    else:
        cohens_d = 0.0

    # 改进百分比
    if stats_a['mean'] != 0:
        improvement_pct = ((stats_b['mean'] - stats_a['mean']) / abs(stats_a['mean']) * 100)
    else:
        improvement_pct = 0.0

    return {
        algorithm_a_name: {
            'mean': stats_a['mean'],
            'std': stats_a['std'],
            'ci': (stats_a['ci_lower'], stats_a['ci_upper']),
            'median': stats_a['median'],
        },
        algorithm_b_name: {
            'mean': stats_b['mean'],
            'std': stats_b['std'],
            'ci': (stats_b['ci_lower'], stats_b['ci_upper']),
            'median': stats_b['median'],
        },
        'test': test_name,
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'cohens_d': float(cohens_d),
        'improvement_pct': float(improvement_pct),
        'n': min_len,
        'is_paired': is_paired,
    }