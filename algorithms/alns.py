"""ALNS自适应大邻域搜索算法"""

import time
import random
import pickle
import math
from typing import List
from collections import defaultdict

from core.config import SystemConfig
from core.entities import Task, RobotState, TaskStatus, Event
from core.environment import SimulationEnvironment


class ALNSScheduler:
    """ALNS调度器"""

    def __init__(self, env: SimulationEnvironment,
                 max_iterations: int = 500,
                 removal_rate: float = 0.3,
                 initial_temp: float = None,
                 cooling_rate: float = None):
        self.env = env
        self.max_iterations = max_iterations

        # 根据规模调整温度 (如果未指定 initial_temp)
        num_tasks = len(env.tasks)
        if initial_temp is not None:
             self.T_start = initial_temp
             # 默认 T_end 没用到，如果指定了 cooling_rate
             self.T_end = 0.1 
        elif num_tasks <= 10:
            self.T_start = 50.0
            self.T_end = 0.1
        elif num_tasks <= 30:
            self.T_start = 30.0
            self.T_end = 0.5
        else:
            self.T_start = 20.0
            self.T_end = 1.0

        self.removal_rate = removal_rate
        
        # 计算冷却速率
        if cooling_rate is not None:
            self.alpha = cooling_rate
        else:
            self.alpha = (self.T_end / self.T_start) ** (1 / max_iterations)
            
        self.T = self.T_start

        # 算子权重
        self.destroy_weights = [1.0, 1.0, 1.0]
        self.repair_weights = [1.0, 1.0, 1.0]

        self.best_solution = None
        self.best_cost = float('inf')

        # 保存初始机器人电量
        self.initial_battery_soc = [robot.battery_soc for robot in env.robots]

        # 迭代历史记录（用于绘图）
        self.iteration_history = {
            'iterations': [],
            'current_cost': [],
            'best_cost': [],
            'temperature': [],
            'destroy_weights': [],
            'repair_weights': [],
            'accepted': [],
            'improved': [],
            'destroy_op': [],
            'repair_op': [],
            'cost_diff': [], # 记录 candidate_cost - current_cost
        }

    def random_removal(self, num_remove: int):
        """随机移除任务"""
        assigned = [t.task_id for t in self.env.tasks.values()
                    if t.assigned_robot is not None]

        if not assigned or num_remove == 0:
            return []

        num_remove = min(num_remove, len(assigned))
        removed = random.sample(assigned, num_remove)

        for tid in removed:
            task = self.env.tasks[tid]
            if task.assigned_robot is not None:
                robot = self.env.robots[task.assigned_robot]
                if tid in robot.carrying_tasks:
                    robot.carrying_tasks.remove(tid)
                task.assigned_robot = None

        return removed

    def worst_removal(self, num_remove: int):
        """移除代价最高的任务"""
        assigned = [(t.task_id, t) for t in self.env.tasks.values()
                    if t.assigned_robot is not None]

        if not assigned or num_remove == 0:
            return []

        assigned.sort(key=lambda x: abs(x[1].dest_floor), reverse=True)

        num_remove = min(num_remove, len(assigned))
        removed = [tid for tid, _ in assigned[:num_remove]]

        for tid in removed:
            task = self.env.tasks[tid]
            if task.assigned_robot is not None:
                robot = self.env.robots[task.assigned_robot]
                if tid in robot.carrying_tasks:
                    robot.carrying_tasks.remove(tid)
                task.assigned_robot = None

        return removed

    def related_removal(self, num_remove: int):
        """移除同一机器人的任务"""
        robot_tasks = defaultdict(list)
        for t in self.env.tasks.values():
            if t.assigned_robot is not None:
                robot_tasks[t.assigned_robot].append(t.task_id)

        if not robot_tasks:
            return []

        target_robot = max(robot_tasks.keys(), key=lambda r: len(robot_tasks[r]))
        tasks = robot_tasks[target_robot]

        num_remove = min(num_remove, len(tasks))
        removed = random.sample(tasks, num_remove)

        for tid in removed:
            task = self.env.tasks[tid]
            robot = self.env.robots[task.assigned_robot]
            if tid in robot.carrying_tasks:
                robot.carrying_tasks.remove(tid)
            task.assigned_robot = None

        return removed

    def smart_greedy_repair(self, removed: List[int]):
        """智能贪心修复"""
        if not removed:
            return True

        removed.sort(key=lambda tid: self.env.tasks[tid].deadline)

        for tid in removed:
            task = self.env.tasks[tid]

            best_robot = None
            best_score = float('inf')

            for robot in self.env.robots:
                if robot.state != RobotState.IDLE:
                    continue
                if robot.needs_charge():
                    continue
                if len(robot.carrying_tasks) >= SystemConfig.ROBOT_CAPACITY:
                    continue
                if not robot.can_complete_mission(task.dest_floor, task.horizontal_distance):
                    continue

                score = (
                    abs(robot.current_floor - SystemConfig.LOBBY_FLOOR) * 0.5 +
                    len(robot.carrying_tasks) * 10 +
                    (100 - robot.battery_soc) * 0.1
                )

                if score < best_score:
                    best_score = score
                    best_robot = robot

            if best_robot:
                task.assigned_robot = best_robot.robot_id
                best_robot.carrying_tasks.append(tid)

        return True

    def regret_repair(self, removed: List[int]):
        """后悔修复"""
        if not removed:
            return True

        while removed:
            best_task = None
            max_regret = -1

            for tid in removed:
                task = self.env.tasks[tid]

                scores = []
                for robot in self.env.robots:
                    if robot.state != RobotState.IDLE:
                        continue
                    if robot.needs_charge():
                        continue
                    if len(robot.carrying_tasks) >= SystemConfig.ROBOT_CAPACITY:
                        continue
                    if not robot.can_complete_mission(task.dest_floor, task.horizontal_distance):
                        continue

                    score = (
                        abs(robot.current_floor - SystemConfig.LOBBY_FLOOR) * 0.5 +
                        len(robot.carrying_tasks) * 10
                    )
                    scores.append((robot, score))

                if len(scores) >= 2:
                    scores.sort(key=lambda x: x[1])
                    regret = scores[1][1] - scores[0][1]
                    if regret > max_regret:
                        max_regret = regret
                        best_task = (tid, scores[0][0])
                elif len(scores) == 1:
                    if 0 > max_regret:
                        max_regret = 0
                        best_task = (tid, scores[0][0])

            if best_task:
                tid, robot = best_task
                task = self.env.tasks[tid]
                task.assigned_robot = robot.robot_id
                robot.carrying_tasks.append(tid)
                removed.remove(tid)
            else:
                break

        return True

    def batch_repair(self, removed: List[int]):
        """批量修复"""
        if not removed:
            return True

        # 允许 ±5 层的容差
        groups = []
        remaining_tasks = removed.copy()
        
        # 简单的聚类逻辑
        while remaining_tasks:
            current_tid = remaining_tasks.pop(0)
            current_task = self.env.tasks[current_tid]
            
            # 尝试加入现有的组
            added_to_group = False
            for group in groups:
                # 取组内第一个任务作为基准楼层
                ref_task = self.env.tasks[group[0]]
                # Δf = 5
                if abs(current_task.dest_floor - ref_task.dest_floor) <= 5:
                    group.append(current_tid)
                    added_to_group = True
                    break
            
            # 如果没法加入现有组，创建新组
            if not added_to_group:
                groups.append([current_tid])

        # 对每一组进行分配
        for task_ids in groups:
            # 获取该组的中心楼层（用于计算 score）
            ref_task = self.env.tasks[task_ids[0]]
            target_floor = ref_task.dest_floor
            
            task_ids.sort(key=lambda tid: self.env.tasks[tid].deadline)

            for tid in task_ids:
                task = self.env.tasks[tid]
                best_robot = None
                best_score = float('inf')

                for robot in self.env.robots:
                    if robot.state != RobotState.IDLE: continue
                    if robot.needs_charge(): continue
                    if len(robot.carrying_tasks) >= SystemConfig.ROBOT_CAPACITY: continue
                    if not robot.can_complete_mission(task.dest_floor, task.horizontal_distance): continue

                    # 评分逻辑
                    score = (
                        abs(robot.current_floor - SystemConfig.LOBBY_FLOOR) * 0.5 -
                        len(robot.carrying_tasks) * 5
                    )

                    if score < best_score:
                        best_score = score
                        best_robot = robot

                if best_robot:
                    task.assigned_robot = best_robot.robot_id
                    best_robot.carrying_tasks.append(tid)

        return True

    def full_evaluate(self, target_env=None, return_metrics=False, enable_log=False):
        """完整评估，支持循环分配"""
        env_to_clone = target_env if target_env else self.env
        eval_env = pickle.loads(pickle.dumps(env_to_clone, -1))

        eval_env.time = 0.0
        eval_env.enable_detailed_log = enable_log

        for task in eval_env.tasks.values():
            task.completion_time = None
            if task.assigned_robot is not None:
                task.status = TaskStatus.ASSIGNED

        for i, robot in enumerate(eval_env.robots):
            robot.state = RobotState.IDLE
            robot.busy_until = 0.0
            robot.current_floor = SystemConfig.LOBBY_FLOOR
            robot.mission = None
            robot.carrying_tasks = []  # 必须清空，否则会保留之前的装载状态导致评估不一致
            robot.battery_soc = self.initial_battery_soc[i]  # 使用真实初始电量
            if hasattr(robot, '_multi_task_counted'):
                delattr(robot, '_multi_task_counted')

        for elevator in eval_env.elevators:
            elevator.current_floor = SystemConfig.LOBBY_FLOOR
            elevator.busy_until = 0.0
            elevator.passengers = []
            elevator.target_floors = []
            elevator.mission = None

        eval_env.total_energy = 0.0
        eval_env.total_wait_time = 0.0
        eval_env.late_count = 0
        eval_env.multi_task_count = 0
        eval_env.batching_count = 0
        eval_env.deadlock_count = 0

        eval_env.elevator_up_queue = [[] for _ in range(SystemConfig.NUM_ELEVATORS)]
        eval_env.elevator_down_queue = [[] for _ in range(SystemConfig.NUM_ELEVATORS)]
        eval_env.floor_waiting_up = {}
        eval_env.floor_waiting_down = {}

        eval_env.event_queue = []

        eval_env.schedule_event(Event(
            time=0.1,
            event_type='elevator_dispatch',
            priority=0,
            data={}
        ))

        # 运行仿真，支持循环分配
        max_sim_time = SystemConfig.get_max_simulation_time(len(eval_env.tasks))
        max_steps = 100000
        steps = 0
        while any(t.completion_time is None for t in eval_env.tasks.values()):
            if eval_env.time > max_sim_time or steps >= max_steps:
                return float('inf')

            # 为空闲机器人分配已规划的任务
            for robot in eval_env.robots:
                if robot.state == RobotState.IDLE:
                    # 检查是否需要充电
                    if robot.needs_charge():
                        eval_env._start_charging(robot.robot_id)
                        continue

                    # 如果没有携带任务，查找分配给该机器人但未携带的任务
                    if not robot.carrying_tasks:
                        # 按deadline排序选择任务
                        candidates = [t for t in eval_env.tasks.values()
                                    if t.assigned_robot == robot.robot_id
                                    and t.completion_time is None
                                    and t.task_id not in robot.carrying_tasks]
                        candidates.sort(key=lambda t: t.deadline)
                        
                        for task in candidates:
                            if len(robot.carrying_tasks) < robot.capacity:
                                if robot.can_complete_mission(task.dest_floor, task.horizontal_distance):
                                    robot.carrying_tasks.append(task.task_id)

                    # 如果有任务就启动
                    if robot.carrying_tasks:
                        eval_env.initiate_robot_mission(robot.robot_id)

            eval_env.step()
            steps += 1

        if return_metrics:
            metrics = eval_env.get_metrics()
            # 如果启用了日志记录，将详细日志也包含在返回的 metrics 中
            if enable_log and hasattr(eval_env, 'detailed_log'):
                metrics['detailed_log'] = eval_env.detailed_log
            return eval_env.compute_objective(), metrics
        return eval_env.compute_objective()

    def accept_solution(self, new_cost: float, current_cost: float):
        """模拟退火接受准则"""
        if new_cost < current_cost:
            return True

        delta = new_cost - current_cost
        prob = math.exp(-delta / self.T)
        return random.random() < prob

    def select_destroy(self):
        """轮盘赌选择destroy算子"""
        total = sum(self.destroy_weights)
        rand = random.uniform(0, total)
        cumsum = 0
        for i, w in enumerate(self.destroy_weights):
            cumsum += w
            if rand <= cumsum:
                return i
        return len(self.destroy_weights) - 1

    def select_repair(self):
        """轮盘赌选择repair算子"""
        total = sum(self.repair_weights)
        rand = random.uniform(0, total)
        cumsum = 0
        for i, w in enumerate(self.repair_weights):
            cumsum += w
            if rand <= cumsum:
                return i
        return len(self.repair_weights) - 1

    def run(self):
        """运行ALNS算法主循环"""
        start_time = time.time()
        enable_log = self.env.enable_detailed_log

        # 初始解：使用贪心算法
        from algorithms.greedy import GreedyScheduler

        initial_env = pickle.loads(pickle.dumps(self.env, -1))
        initial_env.enable_detailed_log = False
        greedy_scheduler = GreedyScheduler(initial_env)
        greedy_scheduler.run()

        initial_metrics = initial_env.get_metrics()
        initial_cost = initial_metrics['objective_value']

        # 提取任务分配关系
        for task in initial_env.tasks.values():
            if task.assigned_robot is not None:
                self.env.tasks[task.task_id].assigned_robot = task.assigned_robot
                self.env.robots[task.assigned_robot].carrying_tasks.append(task.task_id)

        # 重置环境状态
        self.env.time = 0.0
        self.env.enable_detailed_log = False
        for task in self.env.tasks.values():
            task.completion_time = None
        for i, robot in enumerate(self.env.robots):
            robot.state = RobotState.IDLE
            robot.busy_until = 0.0
            robot.current_floor = SystemConfig.LOBBY_FLOOR
            robot.battery_soc = self.initial_battery_soc[i]  # 重置电量，确保 Repair 操作能正常工作

        self.best_solution = pickle.loads(pickle.dumps(self.env, -1))
        
        # 使用 full_evaluate 计算初始 cost，确保与迭代中的评估标准一致
        initial_cost = self.full_evaluate()
        self.best_cost = initial_cost

        current_solution = pickle.loads(pickle.dumps(self.env, -1))
        current_cost = initial_cost

        print(f"[ALNS] 初始解: {initial_cost:.1f}")

        # 定义算子
        destroy_ops = [self.random_removal, self.worst_removal, self.related_removal]
        repair_ops = [self.smart_greedy_repair, self.regret_repair, self.batch_repair]

        # ALNS主循环
        improvements = 0
        for iteration in range(self.max_iterations):
            # 复制当前解
            candidate = pickle.loads(pickle.dumps(current_solution, -1))
            self.env = candidate

            # 选择算子
            destroy_idx = self.select_destroy()
            repair_idx = self.select_repair()

            # 动态调整移除率
            progress = iteration / self.max_iterations
            num_tasks = len(self.env.tasks)
            if num_tasks <= 10:
                base_rate = 0.4
            elif num_tasks <= 30:
                base_rate = 0.25
            else:
                base_rate = 0.20

            if progress > 0.7:
                base_rate *= 0.7

            total_assigned = len([t for t in self.env.tasks.values()
                                  if t.assigned_robot is not None])
            num_remove = max(1, int(total_assigned * base_rate))

            # Destroy & Repair
            removed = destroy_ops[destroy_idx](num_remove)

            if removed:
                repair_ops[repair_idx](removed)

            # 评估候选解
            candidate_cost = self.full_evaluate()

            # 接受判断
            cost_diff = candidate_cost - current_cost
            is_accepted = self.accept_solution(candidate_cost, current_cost)
            is_improved = False
            
            if is_accepted:
                current_solution = pickle.loads(pickle.dumps(self.env, -1))
                current_cost = candidate_cost

                # 更新全局最优解
                if candidate_cost < self.best_cost:
                    self.best_solution = pickle.loads(pickle.dumps(self.env, -1))
                    self.best_cost = candidate_cost
                    improvements += 1
                    is_improved = True
                    
                    if improvements <= 3:
                        improvement_pct = (initial_cost - self.best_cost) / initial_cost * 100
                        print(f"[ALNS] 迭代{iteration}: 最优={self.best_cost:.1f}, 改进={improvement_pct:.1f}%")

            # 记录迭代历史
            self.iteration_history['iterations'].append(iteration)
            self.iteration_history['current_cost'].append(current_cost)
            self.iteration_history['best_cost'].append(self.best_cost)
            self.iteration_history['temperature'].append(self.T)
            self.iteration_history['destroy_weights'].append(self.destroy_weights.copy())
            self.iteration_history['repair_weights'].append(self.repair_weights.copy())
            self.iteration_history['accepted'].append(is_accepted)
            self.iteration_history['improved'].append(is_improved)
            self.iteration_history['destroy_op'].append(destroy_idx)
            self.iteration_history['repair_op'].append(repair_idx)
            self.iteration_history['cost_diff'].append(cost_diff)

            # 权重更新规则:
            # 1. 找到新最优解 -> * 1.1
            # 2. 接受解但非最优 -> * 1.05
            # 3. 拒绝解 -> 不变
            if is_improved:
                # 情况1：新全局最优
                self.destroy_weights[destroy_idx] *= 1.1
                self.repair_weights[repair_idx] *= 1.1
            elif is_accepted:
                # 情况2：接受但非全局最优 (包括局部改进和SA接受的劣解)
                self.destroy_weights[destroy_idx] *= 1.05
                self.repair_weights[repair_idx] *= 1.05
            # 情况3：拒绝 (不做操作)

            # 降温
            self.T *= self.alpha

        # 最终验证：用最优解运行完整仿真
        self.env = self.best_solution

        # 使用 full_evaluate 进行最终的一致性评估
        # 这保证了最终输出的 Metrics 与搜索过程中使用的 Cost 完全一致
        final_cost, metrics = self.full_evaluate(self.best_solution, return_metrics=True, enable_log=enable_log)
        
        # 强制使用搜索时验证过的 objective_value，避免随机性导致的差异
        metrics['objective_value'] = self.best_cost
        
        metrics['algorithm'] = 'ALNS'
        metrics['run_time'] = time.time() - start_time
        metrics['iterations'] = self.max_iterations
        metrics['iteration_history'] = self.iteration_history

        improvement_pct = (initial_cost - self.best_cost) / initial_cost * 100 if initial_cost > 0 else 0
        print(f"[ALNS] 完成，最优解={self.best_cost:.1f}，改进={improvement_pct:.1f}%，"
              f"改进次数={improvements}")

        return metrics


def run_alns(tasks: List[Task], max_iterations: int = None, seed: int = None, enable_log: bool = False):
    """运行 ALNS 算法"""
    env = SimulationEnvironment()
    env.reset(tasks, seed=seed, enable_log=enable_log)

    # 默认参数配置
    # 如果用户没有指定迭代次数，则根据题目规模自动调整
    if max_iterations is None:
        if len(tasks) <= 10:
            max_iterations = 300
        else:
            max_iterations = 500

    # 根据规模调整移除率
    if len(tasks) <= 10:
        removal_rate = 0.4
    elif len(tasks) <= 30:
        removal_rate = 0.25
    else:
        removal_rate = 0.20

    scheduler = ALNSScheduler(
        env,
        max_iterations=max_iterations,
        removal_rate=removal_rate
    )

    return scheduler.run()