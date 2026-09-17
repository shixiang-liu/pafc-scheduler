"""任务生成器"""

import random
from typing import List

from core.config import SystemConfig
from core.entities import Task


def generate_tasks(num_tasks: int, seed: int = None) -> List[Task]:
    """
    生成任务集

    楼层分布：
    - 60%: 办公楼层（20-100层）
    - 30%: 高层（100-118层）
    - 10%: 低层（1-20层）
    """
    if seed is not None:
        random.seed(seed)

    tasks = []

    for i in range(num_tasks):
        # 楼层分布
        rand = random.random()
        if rand < 0.6:
            dest_floor = random.randint(20, 100)
        elif rand < 0.9:
            dest_floor = random.randint(100, 118)
        else:
            dest_floor = random.randint(1, 20)

        # 水平距离
        horizontal_distance = max(10.0, min(
            SystemConfig.MAX_HORIZONTAL_DISTANCE,
            random.gauss(30.0, 10.0)
        ))

        # 任务释放时间
        if i < num_tasks * 0.1:
            release_time = random.uniform(0, 20)
        else:
            release_time = 0.0

        tasks.append(Task(
            task_id=i,
            dest_floor=dest_floor,
            release_time=release_time,
            deadline=SystemConfig.DEFAULT_DEADLINE,
            horizontal_distance=horizontal_distance
        ))

    return tasks