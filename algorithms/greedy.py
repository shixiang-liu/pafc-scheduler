"""贪心调度算法 - EDF策略（最早截止时间优先）"""

import time
import pickle
from typing import List

from core.config import SystemConfig
from core.entities import Task, RobotState
from core.environment import SimulationEnvironment, Event


class GreedyScheduler:
    """贪心调度器"""

    def __init__(self, env: SimulationEnvironment):
        self.env = env

    def select_task(self, robot_id: int, available_tasks: List[int]):
        """为机器人选择deadline最早的可行任务"""
        if not available_tasks:
            return None
        robot = self.env.robots[robot_id]
        # 过滤出电量足够的任务
        feasible_tasks = []
        for tid in available_tasks:
            task = self.env.tasks[tid]
            if robot.can_complete_mission(task.dest_floor, task.horizontal_distance):
                feasible_tasks.append(tid)
        if not feasible_tasks:
            return None
        # 返回deadline最早的任务
        return min(feasible_tasks, key=lambda tid: self.env.tasks[tid].deadline)

    def run(self):
        """运行贪心调度算法"""
        start_time = time.time()
        max_sim_time = SystemConfig.get_max_simulation_time(len(self.env.tasks))
        max_steps = 100000
        steps = 0

        while steps < max_steps:
            steps += 1

            # 获取可分配的任务
            available_tasks = [
                task.task_id
                for task in self.env.tasks.values()
                if task.assigned_robot is None
                and task.completion_time is None
                and task.release_time <= self.env.time
            ]

            # 为每个空闲机器人分配任务或充电
            for robot in self.env.robots:
                if robot.state != RobotState.IDLE:
                    continue

                # 检查是否需要充电
                if robot.needs_charge():
                    self.env._start_charging(robot.robot_id)
                    continue

                # 尝试分配任务
                if available_tasks:
                    task_id = self.select_task(robot.robot_id, available_tasks)
                    if task_id is None:
                        continue

                    success = self.env.assign_task_to_robot(robot.robot_id, task_id)
                    if success:
                        self.env.initiate_robot_mission(robot.robot_id)
                        if task_id in available_tasks:
                            available_tasks.remove(task_id)

            # 推进仿真
            self.env.step()

            # 检查是否完成
            if all(task.completion_time is not None for task in self.env.tasks.values()):
                break

            if self.env.time > max_sim_time:
                break

        metrics = self.env.get_metrics()
        metrics['algorithm'] = 'Greedy-EDF'
        metrics['run_time'] = time.time() - start_time

        if self.env.enable_detailed_log:
            metrics['detailed_log'] = self.env.detailed_log

        return metrics


def unified_evaluate(env: SimulationEnvironment, initial_battery_soc: List[float], enable_log: bool = False):
    """
    统一的评估函数，用于公平比较不同算法
    基于已有的 task.assigned_robot 分配关系，用统一的执行逻辑进行评估
    """
    eval_env = pickle.loads(pickle.dumps(env, -1))
    
    eval_env.time = 0.0
    eval_env.enable_detailed_log = enable_log
    
    for task in eval_env.tasks.values():
        task.completion_time = None
        if task.assigned_robot is not None:
            from core.entities import TaskStatus
            task.status = TaskStatus.ASSIGNED
    
    for i, robot in enumerate(eval_env.robots):
        robot.state = RobotState.IDLE
        robot.busy_until = 0.0
        robot.current_floor = SystemConfig.LOBBY_FLOOR
        robot.mission = None
        robot.carrying_tasks = []
        robot.battery_soc = initial_battery_soc[i]
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
    
    max_sim_time = SystemConfig.get_max_simulation_time(len(eval_env.tasks))
    max_steps = 100000
    steps = 0
    
    while any(t.completion_time is None for t in eval_env.tasks.values()):
        if eval_env.time > max_sim_time or steps >= max_steps:
            break
        
        for robot in eval_env.robots:
            if robot.state == RobotState.IDLE:
                if robot.needs_charge():
                    eval_env._start_charging(robot.robot_id)
                    continue
                
                if not robot.carrying_tasks:
                    candidates = [t for t in eval_env.tasks.values()
                                if t.assigned_robot == robot.robot_id
                                and t.completion_time is None
                                and t.task_id not in robot.carrying_tasks]
                    candidates.sort(key=lambda t: t.deadline)
                    
                    for task in candidates:
                        if len(robot.carrying_tasks) < robot.capacity:
                            if robot.can_complete_mission(task.dest_floor, task.horizontal_distance):
                                robot.carrying_tasks.append(task.task_id)
                
                if robot.carrying_tasks:
                    eval_env.initiate_robot_mission(robot.robot_id)
        
        eval_env.step()
        steps += 1
    
    metrics = eval_env.get_metrics()
    if enable_log:
        metrics['detailed_log'] = eval_env.detailed_log
    
    return metrics


def run_greedy(tasks: List[Task], seed: int = None, enable_log: bool = False):
    """运行贪心算法"""
    env = SimulationEnvironment()
    env.reset(tasks, seed=seed, enable_log=enable_log)
    
    # 保存初始电量
    initial_battery_soc = [robot.battery_soc for robot in env.robots]
    
    scheduler = GreedyScheduler(env)
    metrics = scheduler.run()
    
    # 使用统一评估函数重新计算 objective_value，确保与 ALNS 可比
    unified_metrics = unified_evaluate(env, initial_battery_soc, enable_log=enable_log)
    metrics['objective_value'] = unified_metrics['objective_value']
    
    return metrics