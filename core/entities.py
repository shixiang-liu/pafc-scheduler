"""实体定义"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "待分配"
    ASSIGNED = "已分配"
    IN_PROGRESS = "执行中"
    COMPLETED = "已完成"


class RobotState(Enum):
    """机器人状态"""
    IDLE = "空闲"
    MOVING_TO_ELEVATOR = "前往电梯"
    WAITING_ELEVATOR = "等待电梯"
    IN_ELEVATOR = "电梯中"
    MOVING_TO_DEST = "前往目的地"
    SERVING = "服务中"
    RETURNING = "返回中"
    CHARGING = "充电中"


@dataclass
class Task:
    """任务实体"""
    task_id: int
    dest_floor: int
    release_time: float
    deadline: float
    horizontal_distance: float
    assigned_robot: Optional[int] = None
    completion_time: Optional[float] = None
    status: TaskStatus = TaskStatus.PENDING


class Robot:
    """机器人实体"""

    def __init__(self, robot_id: int, capacity: int, battery_capacity: float):
        self.robot_id = robot_id
        self.capacity = capacity
        self.battery_capacity = battery_capacity
        self.battery_soc = battery_capacity
        self.state = RobotState.IDLE
        self.current_floor = -2  # B2层
        self.busy_until = 0.0
        self.carrying_tasks: List[int] = []
        self.mission = None

    def needs_charge(self):
        """检查是否需要充电"""
        from core.config import SystemConfig
        return self.battery_soc < SystemConfig.BATTERY_LOW_THRESHOLD

    def can_complete_mission(self, dest_floor: int, horizontal_distance: float):
        """检查电量是否足够完成任务"""
        from core.config import SystemConfig

        # 估算往返时间（分钟）
        horizontal_time = (horizontal_distance * 2) / SystemConfig.ROBOT_SPEED / 60.0
        service_time = SystemConfig.SERVICE_TIME / 60.0

        # 估算能耗
        estimated_energy = (horizontal_time + service_time) * SystemConfig.POWER_MOVING

        # 留30%安全余量
        return self.battery_soc >= estimated_energy * 1.3


class Elevator:
    """电梯实体"""

    def __init__(self, elevator_id: int, capacity: int):
        self.elevator_id = elevator_id
        self.capacity = capacity
        self.current_floor = -2  # B2层
        self.busy_until = 0.0
        self.passengers: List[int] = []
        self.target_floors: List[int] = []
        self.mission = None

    @staticmethod
    def travel_time(from_floor: int, to_floor: int):
        """计算电梯移动时间（包含开关门）"""
        from core.config import SystemConfig

        if from_floor == to_floor:
            return 0.0

        floor_diff = abs(to_floor - from_floor)
        height_diff = floor_diff * SystemConfig.FLOOR_HEIGHT

        # 根据方向选择速度
        if to_floor > from_floor:
            travel_time = height_diff / SystemConfig.ELEVATOR_SPEED_UP
        else:
            travel_time = height_diff / SystemConfig.ELEVATOR_SPEED_DOWN

        return travel_time + SystemConfig.DOOR_TIME


@dataclass
class Event:
    """事件"""
    time: float
    event_type: str
    priority: int
    data: dict

    def __lt__(self, other):
        """用于优先队列排序"""
        if self.time != other.time:
            return self.time < other.time
        return self.priority < other.priority