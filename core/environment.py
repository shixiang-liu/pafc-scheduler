"""离散事件仿真环境"""

import heapq
import random
from typing import List, Optional, Dict

from core.config import SystemConfig
from core.entities import Task, Robot, Elevator, Event, RobotState, TaskStatus


class SimulationEnvironment:
    """离散事件仿真环境"""

    def __init__(self):
        self.time = 0.0
        self.robots: List[Robot] = []
        self.elevators: List[Elevator] = []
        self.tasks: Dict[int, Task] = {}
        self.event_queue: List[Event] = []

        # 电梯队列：elevator_up_queue[elev_id] = [(robot_id, wait_floor), ...]
        # 上行队列存储(机器人ID, 等待楼层)元组
        self.elevator_up_queue = []
        self.elevator_down_queue = []

        # 统计指标
        self.total_energy = 0.0
        self.total_wait_time = 0.0
        self.late_count = 0
        self.multi_task_count = 0
        self.batching_count = 0
        self.deadlock_count = 0

        # 日志
        self.enable_detailed_log = False
        self.detailed_log = []

    def reset(self, tasks: List[Task], seed: Optional[int] = None, enable_log: bool = False):
        """重置环境"""
        if seed is not None:
            random.seed(seed)

        self.time = 0.0
        self.enable_detailed_log = enable_log
        self.detailed_log = []

        # 初始化机器人
        self.robots = []
        for i in range(SystemConfig.NUM_ROBOTS):
            robot = Robot(i, SystemConfig.ROBOT_CAPACITY, SystemConfig.BATTERY_CAPACITY)
            robot.battery_soc = random.uniform(30.0, 50.0)  # 午高峰前已工作过
            self.robots.append(robot)

        # 初始化电梯
        self.elevators = [
            Elevator(i, SystemConfig.ELEVATOR_CAPACITY)
            for i in range(SystemConfig.NUM_ELEVATORS)
        ]

        # 初始化电梯队列
        self.elevator_up_queue = [[] for _ in range(SystemConfig.NUM_ELEVATORS)]
        self.elevator_down_queue = [[] for _ in range(SystemConfig.NUM_ELEVATORS)]
        # 按楼层索引的等待队列：{楼层: [机器人ID列表]}
        # 用于支持中间楼层接载机器人
        self.floor_waiting_up = {}    # 等待上行
        self.floor_waiting_down = {}  # 等待下行

        # 深拷贝task对象，避免多个算法共享
        self.tasks = {}
        for task in tasks:
            new_task = Task(
                task_id=task.task_id,
                dest_floor=task.dest_floor,
                release_time=task.release_time,
                deadline=task.deadline,
                horizontal_distance=task.horizontal_distance
            )
            self.tasks[task.task_id] = new_task

        # 重置统计
        self.total_energy = 0.0
        self.total_wait_time = 0.0
        self.late_count = 0
        self.multi_task_count = 0
        self.batching_count = 0
        self.deadlock_count = 0

        # 重置事件队列
        self.event_queue = []

        # 添加初始电梯调度事件
        self.schedule_event(Event(
            time=0.1,
            event_type="elevator_dispatch",
            priority=0,
            data={}
        ))

    def schedule_event(self, event: Event):
        """调度事件"""
        heapq.heappush(self.event_queue, event)

    def step(self):
        """执行一个仿真步骤"""
        if not self.event_queue:
            self.time += 1.0
            self.schedule_event(Event(
                time=self.time,
                event_type="elevator_dispatch",
                priority=0,
                data={}
            ))
            return

        event = heapq.heappop(self.event_queue)
        self.time = event.time

        # 处理事件
        if event.event_type == "elevator_dispatch":
            self._dispatch_elevators()
            self.schedule_event(Event(
                time=self.time + 1.5,
                event_type="elevator_dispatch",
                priority=0,
                data={}
            ))

        elif event.event_type == "robot_finish_task":
            self._robot_finish_task(event.data['robot_id'])

        elif event.event_type == "elevator_arrive":
            self._elevator_arrive(event.data['elevator_id'])

        elif event.event_type == "robot_charge_complete":
            self._robot_charge_complete(event.data['robot_id'])

        elif event.event_type == "robot_arrive_elevator":
            self._robot_arrive_elevator(event.data)

    def _log(self, event_type: str, message: str, data: dict = None):
        """记录详细日志"""
        if self.enable_detailed_log:
            log_entry = {
                'time': round(self.time, 2),
                'event': event_type,
                'message': message
            }
            if data:
                log_entry['data'] = data
            self.detailed_log.append(log_entry)

    def assign_task_to_robot(self, robot_id: int, task_id: int):
        """分配任务给机器人"""
        robot = self.robots[robot_id]
        task = self.tasks[task_id]

        if len(robot.carrying_tasks) >= robot.capacity:
            return False
        if task.assigned_robot is not None:
            return False
        if not robot.can_complete_mission(task.dest_floor, task.horizontal_distance):
            return False

        task.assigned_robot = robot_id
        task.status = TaskStatus.ASSIGNED
        robot.carrying_tasks.append(task_id)

        self._log('ASSIGN', f'任务T{task_id}分配给机器人R{robot_id}', {
            'task_id': task_id,
            'robot_id': robot_id,
            'dest_floor': task.dest_floor,
            'deadline': task.deadline,
            'robot_battery': round(robot.battery_soc, 1)
        })

        return True

    def initiate_robot_mission(self, robot_id: int):
        """启动机器人任务，机器人从卸货区移动到电梯等候区"""
        robot = self.robots[robot_id]

        if not robot.carrying_tasks or robot.state != RobotState.IDLE:
            return False

        # 统计多任务
        if len(robot.carrying_tasks) >= 2 and not hasattr(robot, '_multi_task_counted'):
            self.multi_task_count += 1
            robot._multi_task_counted = True

        # 按deadline排序，优先执行紧急任务
        tasks = [self.tasks[tid] for tid in robot.carrying_tasks]
        tasks.sort(key=lambda t: t.deadline)
        robot.carrying_tasks = [t.task_id for t in tasks]

        target_floor = tasks[0].dest_floor

        # 选择电梯：计算每个电梯的预估负载（队列中人数 + 正在前往的人数）
        # 使用轮询打破平局，确保负载均衡
        if target_floor > SystemConfig.LOBBY_FLOOR:
            # 统计正在前往各电梯的机器人数（通过事件队列估算）
            pending_count = [0] * SystemConfig.NUM_ELEVATORS
            for event in self.event_queue:
                if event.event_type == 'robot_arrive_elevator' and event.data.get('is_up'):
                    pending_count[event.data['elevator_id']] += 1
            
            # 选择总负载最小的电梯
            loads = [len(self.elevator_up_queue[i]) + pending_count[i] 
                    for i in range(SystemConfig.NUM_ELEVATORS)]
            best_elev = min(range(SystemConfig.NUM_ELEVATORS), key=lambda i: (loads[i], i))
        else:
            pending_count = [0] * SystemConfig.NUM_ELEVATORS
            for event in self.event_queue:
                if event.event_type == 'robot_arrive_elevator' and not event.data.get('is_up'):
                    pending_count[event.data['elevator_id']] += 1
            
            loads = [len(self.elevator_down_queue[i]) + pending_count[i] 
                    for i in range(SystemConfig.NUM_ELEVATORS)]
            best_elev = min(range(SystemConfig.NUM_ELEVATORS), key=lambda i: (loads[i], i))

        # 机器人需要从卸货区移动到电梯等候区（耗时20秒）
        move_time = SystemConfig.DEPOT_TO_ELEVATOR
        arrive_time = self.time + move_time

        self._log('ROBOT_MOVE', f'R{robot_id}从卸货区出发前往电梯E{best_elev}，携带任务{robot.carrying_tasks}', {
            'robot_id': robot_id,
            'tasks': robot.carrying_tasks.copy(),
            'target_floor': target_floor,
            'elevator_id': best_elev,
            'move_time': move_time,
            'arrive_time': round(arrive_time, 1)
        })

        # 调度到达电梯等候区事件
        self.schedule_event(Event(
            time=arrive_time,
            event_type='robot_arrive_elevator',
            priority=1,
            data={
                'robot_id': robot_id,
                'elevator_id': best_elev,
                'is_up': target_floor > SystemConfig.LOBBY_FLOOR,
                'wait_floor': SystemConfig.LOBBY_FLOOR  # 从B2层出发
            }
        ))

        robot.state = RobotState.WAITING_ELEVATOR
        return True

    def _robot_arrive_elevator(self, data: dict):
        """机器人到达电梯等候区，加入等待队列"""
        robot_id = data['robot_id']
        elev_id = data['elevator_id']
        is_up = data['is_up']

        robot = self.robots[robot_id]

        # 加入对应电梯队列
        if is_up:
            self.elevator_up_queue[elev_id].append(robot_id)
            queue_len = len(self.elevator_up_queue[elev_id])
        else:
            self.elevator_down_queue[elev_id].append(robot_id)
            queue_len = len(self.elevator_down_queue[elev_id])

        self._log('ROBOT_WAIT', f'R{robot_id}到达电梯E{elev_id}等候区，当前队列{queue_len}人', {
            'robot_id': robot_id,
            'elevator_id': elev_id,
            'queue_length': queue_len,
            'tasks': robot.carrying_tasks.copy()
        })

    def _select_down_elevator(self):
        """选择下行电梯（考虑负载均衡）"""
        # 统计正在前往各电梯的机器人数
        pending_count = [0] * SystemConfig.NUM_ELEVATORS
        for event in self.event_queue:
            if event.event_type == 'robot_arrive_elevator' and not event.data.get('is_up'):
                pending_count[event.data['elevator_id']] += 1
        
        loads = [len(self.elevator_down_queue[i]) + pending_count[i] 
                for i in range(SystemConfig.NUM_ELEVATORS)]
        return min(range(SystemConfig.NUM_ELEVATORS), key=lambda i: (loads[i], i))

    def _select_up_elevator(self):
        """选择上行电梯（考虑负载均衡）"""
        # 统计正在前往各电梯的机器人数
        pending_count = [0] * SystemConfig.NUM_ELEVATORS
        for event in self.event_queue:
            if event.event_type == 'robot_arrive_elevator' and event.data.get('is_up'):
                pending_count[event.data['elevator_id']] += 1
        
        loads = [len(self.elevator_up_queue[i]) + pending_count[i] 
                for i in range(SystemConfig.NUM_ELEVATORS)]
        return min(range(SystemConfig.NUM_ELEVATORS), key=lambda i: (loads[i], i))

    def _dispatch_elevators(self):
        """电梯调度（支持中间楼层接载）"""
        for elev_id, elevator in enumerate(self.elevators):
            if elevator.busy_until > self.time:
                continue

            if elevator.passengers:
                continue

            # 优先处理B2层队列
            if self.elevator_up_queue[elev_id]:
                self._load_elevator(elev_id, is_up=True)
            elif self.elevator_down_queue[elev_id]:
                self._load_elevator(elev_id, is_up=False)
            else:
                # 空闲电梯检查是否有中间楼层等待的机器人
                waiting_up = [f for f, robots in self.floor_waiting_up.items() if robots]
                waiting_down = [f for f, robots in self.floor_waiting_down.items() if robots]
                
                current_floor = elevator.current_floor
                
                if waiting_up:
                    # 有上行等待
                    pickup_floors = sorted(waiting_up)
                    elevator.target_floors = pickup_floors
                    
                    next_floor = elevator.target_floors[0]
                    travel_time = Elevator.travel_time(current_floor, next_floor)
                    elevator.busy_until = self.time + travel_time
                    
                    floor_name = f'{current_floor}F' if current_floor > 0 else f'B{abs(current_floor)}'
                    self._log('ELEVATOR_DISPATCH_PICKUP', 
                        f'电梯E{elev_id}从{floor_name}出发接载上行等待→{elevator.target_floors}', {
                        'elevator_id': elev_id,
                        'current_floor': current_floor,
                        'target_floors': elevator.target_floors.copy()
                    })
                    
                    self.schedule_event(Event(
                        time=elevator.busy_until,
                        event_type="elevator_arrive",
                        priority=1,
                        data={'elevator_id': elev_id}
                    ))
                elif waiting_down:
                    # 有下行等待（从高到低排序）
                    pickup_floors = sorted(waiting_down, reverse=True)
                    elevator.target_floors = pickup_floors + [SystemConfig.LOBBY_FLOOR]
                    
                    next_floor = elevator.target_floors[0]
                    travel_time = Elevator.travel_time(current_floor, next_floor)
                    elevator.busy_until = self.time + travel_time
                    
                    floor_name = f'{current_floor}F' if current_floor > 0 else f'B{abs(current_floor)}'
                    self._log('ELEVATOR_DISPATCH_DOWN', 
                        f'电梯E{elev_id}从{floor_name}出发接载下行等待→{elevator.target_floors}', {
                        'elevator_id': elev_id,
                        'current_floor': current_floor,
                        'target_floors': elevator.target_floors.copy()
                    })
                    
                    self.schedule_event(Event(
                        time=elevator.busy_until,
                        event_type="elevator_arrive",
                        priority=1,
                        data={'elevator_id': elev_id}
                    ))

    def _load_elevator(self, elev_id: int, is_up: bool):
        """装载电梯（支持多层行程）"""
        elevator = self.elevators[elev_id]
        current_floor = elevator.current_floor
        
        if is_up:
            queue = self.elevator_up_queue[elev_id]
        else:
            queue = self.elevator_down_queue[elev_id]
        
        # 装载机器人
        num_load = min(len(queue), elevator.capacity - len(elevator.passengers))
        loaded = queue[:num_load]
        
        if is_up:
            self.elevator_up_queue[elev_id] = queue[num_load:]
        else:
            self.elevator_down_queue[elev_id] = queue[num_load:]

        if not loaded:
            return  # 没有装载任何机器人

        # 统计拼梯
        total_passengers = len(elevator.passengers) + len(loaded)
        if total_passengers >= 2 and len(elevator.passengers) == 0:
            self.batching_count += 1

        elevator.passengers.extend(loaded)
        for rid in loaded:
            self.robots[rid].state = RobotState.IN_ELEVATOR

        # 收集目标楼层 - 收集所有乘客的所有任务楼层（多层行程核心）
        floors = set()
        if is_up:
            for rid in elevator.passengers:
                robot = self.robots[rid]
                for tid in robot.carrying_tasks:
                    task = self.tasks[tid]
                    if task.completion_time is None:
                        floors.add(task.dest_floor)
            elevator.target_floors = sorted(floors)  # 按楼层从低到高排序
        else:
            elevator.target_floors = [SystemConfig.LOBBY_FLOOR]

        # 记录电梯装载日志
        robot_info = ', '.join([f'R{rid}' for rid in loaded])
        direction = '上行' if is_up else '下行'
        floor_name = f'{current_floor}F' if current_floor > 0 else f'B{abs(current_floor)}'
        is_multi_floor = len(elevator.target_floors) > 1
        trip_type = '多层行程' if is_multi_floor else '单层行程'
        self._log('ELEVATOR_LOAD', 
            f'电梯E{elev_id}在{floor_name}{direction}装载[{robot_info}]，{trip_type}→{elevator.target_floors}', {
            'elevator_id': elev_id,
            'load_floor': current_floor,
            'direction': direction,
            'loaded_robots': loaded.copy(),
            'all_passengers': elevator.passengers.copy(),
            'target_floors': elevator.target_floors.copy(),
            'is_multi_floor_trip': is_multi_floor
        })

        # 前往第一个目标
        if elevator.target_floors:
            next_floor = elevator.target_floors[0]
            travel_time = Elevator.travel_time(current_floor, next_floor)
            elevator.busy_until = self.time + travel_time

            self.schedule_event(Event(
                time=elevator.busy_until,
                event_type="elevator_arrive",
                priority=1,
                data={'elevator_id': elev_id}
            ))

    def _elevator_arrive(self, elevator_id: int):
        """电梯到达楼层（支持多层行程）"""
        elevator = self.elevators[elevator_id]

        if not elevator.target_floors:
            return

        current_floor = elevator.target_floors[0]
        elevator.current_floor = current_floor
        elevator.target_floors = elevator.target_floors[1:]

        floor_name = f'{current_floor}F' if current_floor > 0 else f'B{abs(current_floor)}'
        self._log('ELEVATOR_ARRIVE', f'电梯E{elevator_id}到达{floor_name}，乘客{len(elevator.passengers)}人', {
            'elevator_id': elevator_id,
            'floor': current_floor,
            'passengers': elevator.passengers.copy(),
            'remaining_floors': elevator.target_floors.copy()
        })

        # 机器人下电梯
        remaining = []
        for rid in elevator.passengers:
            robot = self.robots[rid]

            # 返回卸货区（B2层）
            if current_floor == SystemConfig.LOBBY_FLOOR:
                robot.current_floor = SystemConfig.LOBBY_FLOOR
                robot.state = RobotState.IDLE
                continue

            # 检查是否有任务在当前楼层
            current_floor_tasks = [
                tid for tid in robot.carrying_tasks
                if self.tasks[tid].dest_floor == current_floor 
                and self.tasks[tid].completion_time is None
            ]

            if current_floor_tasks:
                # 在当前楼层有任务，执行第一个（任务列表已按deadline排序）
                task_id = current_floor_tasks[0]
                task = self.tasks[task_id]
                
                robot.current_floor = current_floor
                robot.state = RobotState.MOVING_TO_DEST

                # 计算任务完成时间
                h_time = task.horizontal_distance / SystemConfig.ROBOT_SPEED
                service_time = SystemConfig.SERVICE_TIME
                total_time = h_time * 2 + service_time

                robot.busy_until = self.time + total_time

                floor_name_log = f'{current_floor}F' if current_floor > 0 else f'B{abs(current_floor)}'
                remaining_count = len(robot.carrying_tasks) - 1
                self._log('ROBOT_DELIVER', 
                    f'机器人R{rid}在{floor_name_log}开始配送T{task.task_id}（剩余{remaining_count}个任务）', {
                    'robot_id': rid,
                    'task_id': task.task_id,
                    'floor': current_floor,
                    'horizontal_distance': task.horizontal_distance,
                    'estimated_duration': round(total_time, 1),
                    'deadline': task.deadline,
                    'time_remaining': round(task.deadline - self.time, 1),
                    'remaining_tasks': remaining_count
                })

                # 扣除能耗
                energy = (h_time * 2 + service_time) / 60.0 * SystemConfig.POWER_MOVING
                robot.battery_soc -= energy
                self.total_energy += energy

                # 调度任务完成事件
                self.schedule_event(Event(
                    time=robot.busy_until,
                    event_type="robot_finish_task",
                    priority=2,
                    data={'robot_id': rid, 'elevator_id': elevator_id}
                ))
            else:
                # 当前楼层没有任务，继续留在电梯
                remaining.append(rid)

        elevator.passengers = remaining

        # 中间楼层接载：接载在当前楼层等待上行的机器人
        if current_floor in self.floor_waiting_up and self.floor_waiting_up[current_floor]:
            waiting_robots = self.floor_waiting_up[current_floor]
            capacity_left = elevator.capacity - len(elevator.passengers)
            
            if capacity_left > 0:
                to_load = waiting_robots[:capacity_left]
                self.floor_waiting_up[current_floor] = waiting_robots[capacity_left:]
                
                for rid in to_load:
                    elevator.passengers.append(rid)
                    self.robots[rid].state = RobotState.IN_ELEVATOR
                    
                    # 将该机器人的目标楼层加入电梯目标（上行）
                    robot = self.robots[rid]
                    for tid in robot.carrying_tasks:
                        task = self.tasks[tid]
                        if task.completion_time is None and task.dest_floor > current_floor:
                            if task.dest_floor not in elevator.target_floors:
                                elevator.target_floors.append(task.dest_floor)
                
                # 重新排序目标楼层（上行从低到高）
                elevator.target_floors = sorted(elevator.target_floors)
                
                floor_name = f'{current_floor}F' if current_floor > 0 else f'B{abs(current_floor)}'
                self._log('ELEVATOR_PICKUP', 
                    f'电梯E{elevator_id}在{floor_name}接载上行[{", ".join(f"R{r}" for r in to_load)}]，目标→{elevator.target_floors}', {
                    'elevator_id': elevator_id,
                    'floor': current_floor,
                    'picked_up': to_load,
                    'all_passengers': elevator.passengers.copy(),
                    'target_floors': elevator.target_floors.copy()
                })

        # 中间楼层接载：接载在当前楼层等待下行的机器人
        if current_floor in self.floor_waiting_down and self.floor_waiting_down[current_floor]:
            waiting_robots = self.floor_waiting_down[current_floor]
            capacity_left = elevator.capacity - len(elevator.passengers)
            
            if capacity_left > 0:
                to_load = waiting_robots[:capacity_left]
                self.floor_waiting_down[current_floor] = waiting_robots[capacity_left:]
                
                for rid in to_load:
                    elevator.passengers.append(rid)
                    self.robots[rid].state = RobotState.IN_ELEVATOR
                    
                    # 将该机器人的目标楼层加入电梯目标（下行）
                    robot = self.robots[rid]
                    for tid in robot.carrying_tasks:
                        task = self.tasks[tid]
                        if task.completion_time is None and task.dest_floor < current_floor:
                            if task.dest_floor not in elevator.target_floors:
                                elevator.target_floors.append(task.dest_floor)
                    
                    # 确保最终回到B2层
                    if SystemConfig.LOBBY_FLOOR not in elevator.target_floors:
                        elevator.target_floors.append(SystemConfig.LOBBY_FLOOR)
                
                # 重新排序目标楼层（下行从高到低）
                elevator.target_floors = sorted(elevator.target_floors, reverse=True)
                
                floor_name = f'{current_floor}F' if current_floor > 0 else f'B{abs(current_floor)}'
                self._log('ELEVATOR_PICKUP_DOWN', 
                    f'电梯E{elevator_id}在{floor_name}接载下行[{", ".join(f"R{r}" for r in to_load)}]，目标→{elevator.target_floors}', {
                    'elevator_id': elevator_id,
                    'floor': current_floor,
                    'picked_up': to_load,
                    'all_passengers': elevator.passengers.copy(),
                    'target_floors': elevator.target_floors.copy()
                })

        # 继续前往下一个楼层
        if elevator.target_floors:
            next_floor = elevator.target_floors[0]
            travel_time = Elevator.travel_time(current_floor, next_floor)
            elevator.busy_until = self.time + travel_time

            self.schedule_event(Event(
                time=elevator.busy_until,
                event_type="elevator_arrive",
                priority=1,
                data={'elevator_id': elevator_id}
            ))
        elif elevator.passengers:
            # 还有乘客但没有目标楼层，这是异常情况
            pass
        else:
            # 电梯空了，检查是否有中间楼层等待接载的机器人
            # 优先检查上行等待，然后检查下行等待
            waiting_up = [(f, robots) for f, robots in self.floor_waiting_up.items() if robots]
            waiting_down = [(f, robots) for f, robots in self.floor_waiting_down.items() if robots]
            
            if waiting_up:
                # 有上行等待，前往接载
                pickup_floors = sorted([f for f, _ in waiting_up])
                elevator.target_floors = pickup_floors
                
                next_floor = elevator.target_floors[0]
                travel_time = Elevator.travel_time(current_floor, next_floor)
                elevator.busy_until = self.time + travel_time
                
                floor_name = f'{current_floor}F' if current_floor > 0 else f'B{abs(current_floor)}'
                self._log('ELEVATOR_CONTINUE_PICKUP', 
                    f'电梯E{elevator_id}从{floor_name}前往接载上行等待→{elevator.target_floors}', {
                    'elevator_id': elevator_id,
                    'current_floor': current_floor,
                    'target_floors': elevator.target_floors.copy()
                })
                
                self.schedule_event(Event(
                    time=elevator.busy_until,
                    event_type="elevator_arrive",
                    priority=1,
                    data={'elevator_id': elevator_id}
                ))
            elif waiting_down:
                # 有下行等待，前往接载（按从高到低排序）
                pickup_floors = sorted([f for f, _ in waiting_down], reverse=True)
                elevator.target_floors = pickup_floors + [SystemConfig.LOBBY_FLOOR]
                
                next_floor = elevator.target_floors[0]
                travel_time = Elevator.travel_time(current_floor, next_floor)
                elevator.busy_until = self.time + travel_time
                
                floor_name = f'{current_floor}F' if current_floor > 0 else f'B{abs(current_floor)}'
                self._log('ELEVATOR_PICKUP_DOWN', 
                    f'电梯E{elevator_id}从{floor_name}前往接载下行等待→{elevator.target_floors}', {
                    'elevator_id': elevator_id,
                    'current_floor': current_floor,
                    'target_floors': elevator.target_floors.copy()
                })
                
                self.schedule_event(Event(
                    time=elevator.busy_until,
                    event_type="elevator_arrive",
                    priority=1,
                    data={'elevator_id': elevator_id}
                ))

    def _robot_finish_task(self, robot_id: int):
        """机器人完成一个任务（支持多层行程）"""
        robot = self.robots[robot_id]

        if not robot.carrying_tasks:
            # 没有任务了，检查是否需要充电
            if robot.needs_charge():
                self._start_charging(robot_id)
            else:
                robot.state = RobotState.IDLE
            return

        # 完成第一个任务（carrying_tasks按deadline排序，但这里按实际执行顺序pop）
        # 找到刚完成的任务（在当前楼层的第一个任务）
        completed_task_id = None
        for tid in robot.carrying_tasks:
            task = self.tasks[tid]
            if task.dest_floor == robot.current_floor and task.completion_time is None:
                completed_task_id = tid
                break
        
        if completed_task_id is None:
            # 如果找不到匹配的，取第一个
            completed_task_id = robot.carrying_tasks[0]
        
        robot.carrying_tasks.remove(completed_task_id)
        task = self.tasks[completed_task_id]
        task.completion_time = self.time
        task.status = TaskStatus.COMPLETED

        # 统计超时
        is_late = self.time > task.deadline
        if is_late:
            self.late_count += 1

        # 记录任务完成日志
        delay = round(self.time - task.deadline, 1) if is_late else 0
        status = f'超时{delay}s' if is_late else '准时'
        self._log('TASK_COMPLETE', f'任务T{completed_task_id}完成({status})', {
            'task_id': completed_task_id,
            'robot_id': robot_id,
            'completion_time': round(self.time, 1),
            'deadline': task.deadline,
            'delay': delay,
            'on_time': not is_late,
            'remaining_tasks': len(robot.carrying_tasks)
        })

        # 还有任务，决定下一步行为
        if robot.carrying_tasks:
            current_floor = robot.current_floor
            
            # 分类剩余任务
            same_floor_tasks = [tid for tid in robot.carrying_tasks 
                if self.tasks[tid].dest_floor == current_floor]
            higher_floor_tasks = [tid for tid in robot.carrying_tasks 
                if self.tasks[tid].dest_floor > current_floor]
            lower_floor_tasks = [tid for tid in robot.carrying_tasks 
                if self.tasks[tid].dest_floor < current_floor]
            
            if same_floor_tasks:
                # 同楼层还有任务，立即执行下一个
                next_task_id = same_floor_tasks[0]
                next_task = self.tasks[next_task_id]
                
                robot.state = RobotState.MOVING_TO_DEST
                h_time = next_task.horizontal_distance / SystemConfig.ROBOT_SPEED
                service_time = SystemConfig.SERVICE_TIME
                total_time = h_time * 2 + service_time
                robot.busy_until = self.time + total_time
                
                floor_name = f'{current_floor}F' if current_floor > 0 else f'B{abs(current_floor)}'
                self._log('ROBOT_DELIVER', 
                    f'R{robot_id}在{floor_name}继续配送T{next_task_id}（同楼层任务）', {
                    'robot_id': robot_id,
                    'task_id': next_task_id,
                    'floor': current_floor,
                    'remaining_tasks': len(robot.carrying_tasks) - 1
                })
                
                # 扣除能耗
                energy = (h_time * 2 + service_time) / 60.0 * SystemConfig.POWER_MOVING
                robot.battery_soc -= energy
                self.total_energy += energy
                
                self.schedule_event(Event(
                    time=robot.busy_until,
                    event_type="robot_finish_task",
                    priority=2,
                    data={'robot_id': robot_id}
                ))
                
            elif higher_floor_tasks:
                # 有更高楼层任务，在当前楼层等待电梯继续上行
                robot.state = RobotState.WAITING_ELEVATOR
                
                # 加入当前楼层的等待队列
                if current_floor not in self.floor_waiting_up:
                    self.floor_waiting_up[current_floor] = []
                self.floor_waiting_up[current_floor].append(robot_id)
                
                next_floors = sorted(set(self.tasks[tid].dest_floor for tid in higher_floor_tasks))
                floor_name = f'{current_floor}F' if current_floor > 0 else f'B{abs(current_floor)}'
                self._log('ROBOT_WAIT_CONTINUE', 
                    f'R{robot_id}在{floor_name}等待电梯继续上行送往{next_floors}', {
                    'robot_id': robot_id,
                    'current_floor': current_floor,
                    'next_floors': next_floors,
                    'remaining_tasks': robot.carrying_tasks.copy()
                })
                
            elif lower_floor_tasks:
                # 有更低楼层任务，在当前楼层等待下行电梯（途中送货）
                robot.state = RobotState.WAITING_ELEVATOR
                
                # 加入当前楼层的下行等待队列
                if current_floor not in self.floor_waiting_down:
                    self.floor_waiting_down[current_floor] = []
                self.floor_waiting_down[current_floor].append(robot_id)
                
                next_floors = sorted(set(self.tasks[tid].dest_floor for tid in lower_floor_tasks), reverse=True)
                floor_name = f'{current_floor}F' if current_floor > 0 else f'B{abs(current_floor)}'
                self._log('ROBOT_WAIT_DOWN', 
                    f'R{robot_id}在{floor_name}等待电梯下行送往{next_floors}', {
                    'robot_id': robot_id,
                    'current_floor': current_floor,
                    'next_floors': next_floors,
                    'remaining_tasks': robot.carrying_tasks.copy()
                })
        else:
            # 所有任务完成，返回B2层
            if robot.current_floor != SystemConfig.LOBBY_FLOOR:
                robot.state = RobotState.WAITING_ELEVATOR
                best_elev = self._select_down_elevator()
                self.elevator_down_queue[best_elev].append(robot_id)
                
                self._log('ROBOT_RETURN', f'R{robot_id}完成所有任务，等待电梯E{best_elev}返回B2层', {
                    'robot_id': robot_id,
                    'current_floor': robot.current_floor,
                    'elevator_id': best_elev
                })
            else:
                if robot.needs_charge():
                    self._start_charging(robot_id)
                else:
                    robot.state = RobotState.IDLE

    def _start_charging(self, robot_id: int):
        """开始充电"""
        robot = self.robots[robot_id]
        robot.state = RobotState.CHARGING

        # 计算充电时间（充到30%）
        charge_needed = SystemConfig.BATTERY_LOW_THRESHOLD - robot.battery_soc
        if charge_needed > 0:
            charge_time = charge_needed / SystemConfig.CHARGING_RATE * 60  # 转换为秒
            robot.busy_until = self.time + charge_time

            # 调度充电完成事件
            self.schedule_event(Event(
                time=robot.busy_until,
                event_type="robot_charge_complete",
                priority=3,
                data={'robot_id': robot_id}
            ))
        else:
            # 电量已够，直接变为IDLE
            robot.state = RobotState.IDLE

    def _robot_charge_complete(self, robot_id: int):
        """机器人充电完成"""
        robot = self.robots[robot_id]
        robot.battery_soc = SystemConfig.BATTERY_LOW_THRESHOLD  # 充到30%
        robot.state = RobotState.IDLE

    def compute_objective(self):
        """计算目标函数值"""
        completion_times = [t.completion_time for t in self.tasks.values()
                           if t.completion_time is not None]
        makespan = max(completion_times) if completion_times else 0.0

        # 超时惩罚（针对已完成但超时的任务）
        total_penalty = sum(
            max(0, t.completion_time - t.deadline)
            for t in self.tasks.values()
            if t.completion_time is not None
        )
        
        # 未完成任务惩罚（关键：确保完成更多任务的解总是更优）
        incomplete_tasks = sum(1 for t in self.tasks.values() if t.completion_time is None)
        incomplete_penalty = incomplete_tasks * 10000.0  # 大惩罚确保完成任务优先

        return (
            SystemConfig.WEIGHT_TIME * makespan +
            SystemConfig.WEIGHT_ENERGY * self.total_energy +
            SystemConfig.WEIGHT_PENALTY * total_penalty +
            incomplete_penalty
        )

    def get_metrics(self):
        """获取统计指标"""
        completed = [t for t in self.tasks.values() if t.completion_time is not None]
        on_time = [t for t in completed if t.completion_time <= t.deadline]

        return {
            'objective_value': self.compute_objective(),
            'makespan': max([t.completion_time for t in completed], default=0.0),
            'total_energy': self.total_energy,
            'completed_tasks': len(completed),
            'total_tasks': len(self.tasks),
            'late_tasks': self.late_count,
            'on_time_rate': len(on_time) / len(self.tasks) if self.tasks else 0.0,
            'batching_count': self.batching_count,
            'multi_task_count': self.multi_task_count,
            'deadlock_count': self.deadlock_count,
        }