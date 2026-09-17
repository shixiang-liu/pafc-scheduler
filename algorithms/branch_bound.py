"""分支限界调度算法"""

import time
import pickle
import heapq
from typing import List, Tuple, Dict

from core.config import SystemConfig
from core.entities import Task, RobotState, TaskStatus, Event
from core.environment import SimulationEnvironment
from algorithms.greedy import GreedyScheduler


# 相近楼层容差，楼层差≤此值可合并为一趟
FLOOR_TOLERANCE = 5


class SearchNode:
    # Best-First搜索节点

    def __init__(self, env_state: bytes, lower_bound: float, depth: int, node_id: int):
        self.env_state = env_state
        self.lower_bound = lower_bound
        self.depth = depth
        self.node_id = node_id

    def __lt__(self, other):
        # 下界小的优先；下界相同时深度大的优先
        if self.lower_bound != other.lower_bound:
            return self.lower_bound < other.lower_bound
        return self.depth > other.depth


class BranchAndBoundScheduler:
    """分支限界调度器"""

    def __init__(self, env: SimulationEnvironment, max_nodes: int = 200000,
                 time_limit: float = 300.0):
        self.env = env
        self.max_nodes = max_nodes
        self.time_limit = time_limit
        self.best_cost = float('inf')
        self.best_solution = None
        self.best_node = 0
        self.nodes_explored = 0
        self.nodes_pruned = 0
        self.nodes_generated = 0  # 生成的总节点数
        self.lb_pruned = 0
        self.infeas_pruned = 0
        self.sym_pruned = 0
        self.start_time = None
        self.improvements = 0
        self.search_complete = False
        self.node_counter = 0
        self.initial_metrics = None
        # 保存初始机器人电量（与ALNS一致）
        self.initial_battery_soc = [robot.battery_soc for robot in env.robots]

    def assign_only(self, env: SimulationEnvironment, robot_id: int, task_id: int) -> bool:
        """只设置任务分配，不加载到机器人（用于评估）"""
        task = env.tasks[task_id]
        if task.assigned_robot is not None:
            return False
        task.assigned_robot = robot_id
        task.status = TaskStatus.ASSIGNED
        return True

    def get_robot_signature(self, robot, env) -> Tuple:
        """获取机器人状态签名，用于对称性剪枝"""
        # 计算分配给该机器人的任务数（而不是carrying_tasks）
        assigned_count = sum(1 for t in env.tasks.values() 
                            if t.assigned_robot == robot.robot_id)
        return (
            robot.state,
            robot.current_floor,
            int(robot.battery_soc / 10),  # 电量分档，减少签名数量
            assigned_count,  # 使用已分配任务数
        )

    def filter_symmetric_robots(self, robots: List, env) -> List:
        """过滤对称机器人，只保留每组等价机器人中ID最小的"""
        # 如果还没有任务被分配，不进行对称性剪枝
        # 因为在初始状态，虽然机器人看起来"等价"，但分配给不同机器人会导致不同的后续分支
        any_assigned = any(t.assigned_robot is not None for t in env.tasks.values())
        if not any_assigned:
            return robots
        
        seen = {}
        result = []

        for robot in robots:
            sig = self.get_robot_signature(robot, env)
            if sig not in seen:
                seen[sig] = robot
                result.append(robot)
            else:
                # 保留ID较小的
                if robot.robot_id < seen[sig].robot_id:
                    result.remove(seen[sig])
                    result.append(robot)
                    seen[sig] = robot
                self.sym_pruned += 1
                self.nodes_pruned += 1

        return result


    def group_floors_by_proximity(self, tasks: List[Task]) -> Dict[int, List[Task]]:
        """按楼层相近性分组任务，楼层差≤FLOOR_TOLERANCE的归为一组"""
        if not tasks:
            return {}

        # 按楼层排序
        sorted_tasks = sorted(tasks, key=lambda t: t.dest_floor)
        groups: Dict[int, List[Task]] = {}

        group_id = 0
        current_group_floor = sorted_tasks[0].dest_floor
        groups[group_id] = [sorted_tasks[0]]

        for task in sorted_tasks[1:]:
            if task.dest_floor - current_group_floor <= FLOOR_TOLERANCE:
                # 属于当前组
                groups[group_id].append(task)
            else:
                # 新建一组
                group_id += 1
                current_group_floor = task.dest_floor
                groups[group_id] = [task]

        return groups

    def estimate_lower_bound(self, env: SimulationEnvironment,
                            pending_tasks: List[Task]) -> float:
        """下界估计，考虑电梯容量和相近楼层捎带"""
        if not pending_tasks:
            # 所有任务已分配，返回0作为下界
            # 实际评估会在搜索时通过run_to_completion完成
            return 0.0

        # 计算当前已完成部分的代价（不含未完成任务惩罚，因为它们还没有机会完成）
        completion_times = [t.completion_time for t in env.tasks.values()
                           if t.completion_time is not None]
        makespan = max(completion_times) if completion_times else 0.0
        
        total_penalty = sum(
            max(0, t.completion_time - t.deadline)
            for t in env.tasks.values()
            if t.completion_time is not None
        )
        
        current_obj = (SystemConfig.WEIGHT_TIME * makespan +
                      SystemConfig.WEIGHT_ENERGY * env.total_energy +
                      SystemConfig.WEIGHT_PENALTY * total_penalty)

        # 先计算并行度
        num_available = max(1, sum(1 for r in env.robots
                                   if r.state != RobotState.CHARGING and not r.needs_charge()))
        num_elevators = SystemConfig.NUM_ELEVATORS
        parallelism = min(num_available, num_elevators * SystemConfig.ELEVATOR_CAPACITY)

        # 按楼层相近性分组
        floor_groups = self.group_floors_by_proximity(pending_tasks)

        total_min_time = 0.0
        total_penalties = 0.0

        for tasks in floor_groups.values():
            max_floor = max(t.dest_floor for t in tasks)
            floor_diff = abs(max_floor - SystemConfig.LOBBY_FLOOR)

            # 电梯往返时间
            vertical_time = (floor_diff * SystemConfig.FLOOR_HEIGHT / SystemConfig.ELEVATOR_SPEED_UP +
                           floor_diff * SystemConfig.FLOOR_HEIGHT / SystemConfig.ELEVATOR_SPEED_DOWN)

            # 该组需要的电梯趟数
            trips = (len(tasks) + SystemConfig.ELEVATOR_CAPACITY - 1) // SystemConfig.ELEVATOR_CAPACITY
            floor_time = vertical_time * trips + SystemConfig.DOOR_TIME * 2 * trips

            # 组内任务的水平配送时间
            for task in tasks:
                h_time = (task.horizontal_distance / SystemConfig.ROBOT_SPEED) * 2
                floor_time += h_time + SystemConfig.SERVICE_TIME

            total_min_time += floor_time

            # 惩罚下界
            for task in tasks:
                queue_wait = len(pending_tasks) * 3.0 / parallelism
                min_task_time = (vertical_time +
                               (task.horizontal_distance / SystemConfig.ROBOT_SPEED) * 2 +
                               SystemConfig.SERVICE_TIME + SystemConfig.DOOR_TIME * 2 + queue_wait)
                earliest = env.time + min_task_time
                if earliest > task.deadline:
                    total_penalties += (earliest - task.deadline)

        parallel_time = total_min_time / parallelism

        # 电梯等待时间估计
        num_trips = sum((len(tasks) + SystemConfig.ELEVATOR_CAPACITY - 1) // SystemConfig.ELEVATOR_CAPACITY
                       for tasks in floor_groups.values())
        wait_time = 8.0 * num_trips / num_elevators

        # 综合下界
        time_lb = parallel_time + wait_time
        # 能耗下界：非常保守
        energy_lb = len(pending_tasks) * 0.01

        # 使用适中的系数平衡性能和最优性
        # Bug修复后，可以适当增加系数
        additional = (SystemConfig.WEIGHT_TIME * time_lb * 0.2 +
                     SystemConfig.WEIGHT_ENERGY * energy_lb * 0.2 +
                     SystemConfig.WEIGHT_PENALTY * total_penalties * 0.2)

        return current_obj + additional

    def run_to_completion(self, env: SimulationEnvironment) -> float:
        """运行仿真直到所有任务完成"""
        # 完整重置环境状态，与ALNS评估逻辑保持一致
        env.time = 0.0
        
        for task in env.tasks.values():
            task.completion_time = None
            if task.assigned_robot is not None:
                task.status = TaskStatus.ASSIGNED
        
        for robot in env.robots:
            robot.state = RobotState.IDLE
            robot.busy_until = 0.0
            robot.current_floor = SystemConfig.LOBBY_FLOOR
            robot.mission = None
            robot.carrying_tasks = []
            robot.battery_soc = self.initial_battery_soc[robot.robot_id]  # 使用真实初始电量
            if hasattr(robot, '_multi_task_counted'):
                delattr(robot, '_multi_task_counted')
        
        for elevator in env.elevators:
            elevator.current_floor = SystemConfig.LOBBY_FLOOR
            elevator.busy_until = 0.0
            elevator.passengers = []
            elevator.target_floors = []
            elevator.mission = None
        
        env.total_energy = 0.0
        env.total_wait_time = 0.0
        env.late_count = 0
        env.multi_task_count = 0
        env.batching_count = 0
        env.deadlock_count = 0
        
        env.elevator_up_queue = [[] for _ in range(SystemConfig.NUM_ELEVATORS)]
        env.elevator_down_queue = [[] for _ in range(SystemConfig.NUM_ELEVATORS)]
        env.floor_waiting_up = {}
        env.floor_waiting_down = {}
        
        env.event_queue = []
        env.schedule_event(Event(
            time=0.1,
            event_type='elevator_dispatch',
            priority=0,
            data={}
        ))
        
        max_sim_time = SystemConfig.get_max_simulation_time(len(env.tasks))
        max_steps = 100000
        steps = 0

        while any(t.completion_time is None for t in env.tasks.values()):
            if env.time > max_sim_time or steps >= max_steps:
                return float('inf')

            for robot in env.robots:
                if robot.state == RobotState.IDLE:
                    if robot.needs_charge():
                        env._start_charging(robot.robot_id)
                        continue

                    # 如果机器人空载，加载已分配但尚未携带的任务
                    if not robot.carrying_tasks:
                        # 按deadline排序选择任务
                        candidates = [t for t in env.tasks.values()
                                    if t.assigned_robot == robot.robot_id
                                    and t.completion_time is None
                                    and t.task_id not in robot.carrying_tasks]
                        candidates.sort(key=lambda t: t.deadline)
                        
                        for task in candidates:
                            if len(robot.carrying_tasks) >= robot.capacity:
                                break
                            if robot.can_complete_mission(task.dest_floor, task.horizontal_distance):
                                robot.carrying_tasks.append(task.task_id)

                    if robot.carrying_tasks:
                        env.initiate_robot_mission(robot.robot_id)

            env.step()
            steps += 1

        return env.compute_objective()

    def best_first_search(self):
        """Best-First搜索"""
        queue: List[SearchNode] = []

        # 初始节点
        initial_env = pickle.dumps(self.env, -1)
        pending = [t for t in self.env.tasks.values()
                  if t.assigned_robot is None and t.completion_time is None]
        initial_lb = self.estimate_lower_bound(self.env, pending)
        self.node_counter += 1
        heapq.heappush(queue, SearchNode(initial_env, initial_lb, 0, self.node_counter))

        while queue:
            if time.time() - self.start_time > self.time_limit:
                break
            if self.nodes_explored >= self.max_nodes:
                break

            node = heapq.heappop(queue)

            # 下界剪枝
            if node.lower_bound >= self.best_cost:
                self.nodes_pruned += 1
                self.lb_pruned += 1
                continue

            self.nodes_explored += 1
            env = pickle.loads(node.env_state)

            pending = [t for t in env.tasks.values()
                      if t.assigned_robot is None and t.completion_time is None]

            # 所有任务已分配
            if not pending:
                cost = self.run_to_completion(env)
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_node = self.nodes_explored
                    self.best_solution = {
                        'assignments': [(t.task_id, t.assigned_robot)
                                      for t in env.tasks.values()
                                      if t.assigned_robot is not None],
                        'cost': cost
                    }
                    self.improvements += 1
                    print(f"[B&B] 找到更优解: {cost:.1f} [节点{self.nodes_explored}]")
                continue
            
            # 任务选择分支：
            # 1. 按Deadline排序的前3个紧急任务
            # 2. 所有与当前空闲机器人处于同一楼层的任务（顺路任务）
            
            # 为确保找到理论最优解，必须尝试所有待分配任务
            # 虽然这会增加搜索空间，但这是精确算法的要求
            pending.sort(key=lambda t: t.deadline)
            task_list = pending
            
            # 为每个候选任务尝试分配
            for task in task_list:
                # 找出与当前任务楼层相近的其他任务
                nearby_tasks = [t for t in pending 
                              if t.task_id != task.task_id and 
                              abs(t.dest_floor - task.dest_floor) <= FLOOR_TOLERANCE]

                # 可用机器人
                robots = [r for r in env.robots
                         if r.state == RobotState.IDLE
                         and not r.needs_charge()
                         and len(r.carrying_tasks) < SystemConfig.ROBOT_CAPACITY
                         and r.can_complete_mission(task.dest_floor, task.horizontal_distance)]

                if not robots:
                    self.nodes_pruned += 1
                    self.infeas_pruned += 1
                    continue

                # 对称性剪枝
                robots = self.filter_symmetric_robots(robots, env)
                
                # 优化机器人排序：优先选择已携带相近楼层任务的机器人
                def robot_priority(r):
                    if r.carrying_tasks:
                        carried_floors = [env.tasks[tid].dest_floor for tid in r.carrying_tasks]
                        min_floor_diff = min(abs(f - task.dest_floor) for f in carried_floors)
                        if min_floor_diff <= FLOOR_TOLERANCE:
                            return (0, min_floor_diff, r.robot_id)
                    return (1, abs(r.current_floor - SystemConfig.LOBBY_FLOOR), r.robot_id)
                
                robots.sort(key=robot_priority)

                for robot in robots:
                    # 只分配任务，不启动执行
                    child_env = pickle.loads(node.env_state)
                    
                    if child_env.assign_task_to_robot(robot.robot_id, task.task_id):
                        
                        child_pending = [t for t in child_env.tasks.values()
                                       if t.assigned_robot is None and t.completion_time is None]
                        child_lb = self.estimate_lower_bound(child_env, child_pending)

                        if child_lb >= self.best_cost:
                            self.nodes_pruned += 1
                            self.lb_pruned += 1
                            self.nodes_generated += 1
                        else:
                            self.node_counter += 1
                            self.nodes_generated += 1
                            heapq.heappush(queue, SearchNode(
                                pickle.dumps(child_env, -1), child_lb, node.depth + 1, self.node_counter))
                    
                    # 策略2：批量分配相近楼层任务
                    if nearby_tasks and len(robot.carrying_tasks) + 2 <= SystemConfig.ROBOT_CAPACITY:
                        batch_env = pickle.loads(node.env_state)
                        batch_robot = batch_env.robots[robot.robot_id]
                        
                        if batch_env.assign_task_to_robot(robot.robot_id, task.task_id):
                            batch_count = 1
                            for nearby in nearby_tasks[:SystemConfig.ROBOT_CAPACITY - len(robot.carrying_tasks) - 1]:
                                if (len(batch_robot.carrying_tasks) < SystemConfig.ROBOT_CAPACITY and
                                    batch_robot.can_complete_mission(nearby.dest_floor, nearby.horizontal_distance)):
                                    if batch_env.assign_task_to_robot(robot.robot_id, nearby.task_id):
                                        batch_count += 1
                            
                            if batch_count > 1:
                                batch_pending = [t for t in batch_env.tasks.values()
                                               if t.assigned_robot is None and t.completion_time is None]
                                batch_lb = self.estimate_lower_bound(batch_env, batch_pending)
                                
                                if batch_lb < self.best_cost:
                                    self.node_counter += 1
                                    self.nodes_generated += 1
                                    heapq.heappush(queue, SearchNode(
                                        pickle.dumps(batch_env, -1), batch_lb, node.depth + batch_count, self.node_counter))

        # 搜索结束后，队列中剩余节点都是下界 >= best_cost 的，统计为隐式剪枝
        remaining_pruned = 0
        while queue:
            node = heapq.heappop(queue)
            if node.lower_bound >= self.best_cost:
                remaining_pruned += 1
        
        if remaining_pruned > 0:
            self.nodes_pruned += remaining_pruned
            self.lb_pruned += remaining_pruned

    def run(self):
        """运行分支限界算法"""
        start_time = time.time()
        self.start_time = start_time

        # 贪心初始解 - 提取分配方案
        greedy_env = pickle.loads(pickle.dumps(self.env, -1))
        GreedyScheduler(greedy_env).run()
        greedy_assignments = [(t.task_id, t.assigned_robot) 
                              for t in greedy_env.tasks.values() 
                              if t.assigned_robot is not None]
        greedy_metrics = greedy_env.get_metrics()
        greedy_direct_cost = greedy_metrics['objective_value']
        
        # 用run_to_completion重新评估Greedy的分配（确保评估逻辑一致）
        eval_env = pickle.loads(pickle.dumps(self.env, -1))
        for tid, rid in greedy_assignments:
            self.assign_only(eval_env, rid, tid)
        greedy_cost = self.run_to_completion(eval_env)
        
        self.best_cost = greedy_cost
        self.best_solution = {
            'assignments': greedy_assignments,
            'cost': self.best_cost
        }
        self.initial_metrics = greedy_metrics
        self.initial_metrics['objective_value'] = greedy_cost  # 更新为一致的评估结果
        
        # 对于较大规模问题，使用ALNS获取更好的上界以加速剪枝
        num_tasks = len(self.env.tasks)
        if num_tasks > 6:
            from algorithms.alns import ALNSScheduler
            alns_env = pickle.loads(pickle.dumps(self.env, -1))
            alns_env.enable_detailed_log = False
            alns_scheduler = ALNSScheduler(alns_env, max_iterations=500)
            alns_result = alns_scheduler.run()
            alns_cost = alns_result['objective_value']
            
            # 直接使用ALNS的评估结果（不重新评估，保持一致性）
            alns_assignments = [(t.task_id, t.assigned_robot) 
                               for t in alns_env.tasks.values() 
                               if t.assigned_robot is not None]
            
            if alns_cost < self.best_cost:
                self.best_cost = alns_cost
                self.best_solution = {
                    'assignments': alns_assignments,
                    'cost': alns_cost
                }
                self.initial_metrics = alns_result
            print(f"[B&B] Greedy初始解: {greedy_cost:.1f}, ALNS初始解: {alns_cost:.1f}, 使用: {self.best_cost:.1f}")
        else:
            print(f"[B&B] Greedy初始解: {greedy_cost:.1f}")

        self.best_first_search()

        # 判断搜索是否完整
        elapsed = time.time() - start_time
        if elapsed >= self.time_limit:
            termination = "时间限制"
        elif self.nodes_explored >= self.max_nodes:
            termination = "节点限制"
        else:
            termination = "完整搜索"
            self.search_complete = True

        # 应用最优解并重新评估
        if self.best_solution:
            task_list = [Task(t.task_id, t.dest_floor, t.release_time,
                             t.deadline, t.horizontal_distance)
                        for t in self.env.tasks.values()]
            self.env.reset(task_list, seed=None, enable_log=self.env.enable_detailed_log)

            for task_id, robot_id in self.best_solution['assignments']:
                self.assign_only(self.env, robot_id, task_id)

            self.run_to_completion(self.env)
            metrics = self.env.get_metrics()
        else:
            metrics = self.initial_metrics

        metrics['algorithm'] = 'B&B'
        metrics['run_time'] = time.time() - start_time
        metrics['nodes_explored'] = self.nodes_explored
        metrics['nodes_generated'] = self.nodes_generated
        metrics['nodes_pruned'] = self.nodes_pruned
        metrics['lb_pruned'] = self.lb_pruned
        metrics['infeas_pruned'] = self.infeas_pruned
        metrics['sym_pruned'] = self.sym_pruned
        metrics['best_node'] = self.best_node
        metrics['search_complete'] = self.search_complete

        improvement = ((greedy_metrics['objective_value'] - metrics['objective_value']) /
                      greedy_metrics['objective_value'] * 100)
        prune_rate = self.nodes_pruned / max(1, self.nodes_generated) * 100

        print(f"[B&B] 完成 | 最优解={metrics['objective_value']:.1f} | "
              f"改进={improvement:.1f}% | 探索={self.nodes_explored} | {termination}", end='')

        if self.nodes_generated > 0:
            print(f" | 生成={self.nodes_generated} 剪枝={prune_rate:.1f}% (LB:{self.lb_pruned} SYM:{self.sym_pruned} INF:{self.infeas_pruned})")
        else:
            print()

        if self.env.enable_detailed_log:
            metrics['detailed_log'] = self.env.detailed_log

        return metrics


def run_bnb(tasks: List[Task], max_nodes: int = 200000,
            time_limit: float = 300.0, seed: int = None, enable_log: bool = False):
    """运行分支限界算法"""
    env = SimulationEnvironment()
    env.reset(tasks, seed=seed, enable_log=enable_log)
    scheduler = BranchAndBoundScheduler(env, max_nodes=max_nodes, time_limit=time_limit)
    return scheduler.run()