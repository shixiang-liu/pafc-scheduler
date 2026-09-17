"""系统配置参数"""


class SystemConfig:
    """系统全局配置"""

    # 建筑参数
    BUILDING_HEIGHT = 599.0
    NUM_FLOORS = 118
    FLOOR_HEIGHT = BUILDING_HEIGHT / NUM_FLOORS
    LOBBY_FLOOR = -2  # B2层作为卸货区

    # 电梯参数
    NUM_ELEVATORS = 2
    ELEVATOR_CAPACITY = 4
    ELEVATOR_SPEED_UP = 10.0
    ELEVATOR_SPEED_DOWN = 6.0
    DOOR_TIME = 15.0

    # 机器人参数
    NUM_ROBOTS = 4
    ROBOT_CAPACITY = 4
    ROBOT_SPEED = 1.0
    SERVICE_TIME = 30.0
    DEPOT_TO_ELEVATOR = 20.0

    # 电池参数
    BATTERY_CAPACITY = 100.0
    BATTERY_LOW_THRESHOLD = 30.0
    POWER_IDLE = 0.02  # %/min
    POWER_MOVING = 0.18  # %/min
    CHARGING_RATE = 0.42  # %/min

    # 任务参数
    DEFAULT_DEADLINE = 420.0
    DEFAULT_RELEASE_TIME = 0.0
    MAX_HORIZONTAL_DISTANCE = 60.0

    # 目标函数权重
    WEIGHT_TIME = 1.0  # 总时长权重
    WEIGHT_ENERGY = 100.0  # 能耗权重
    WEIGHT_PENALTY = 50.0  # 超时惩罚权重

    @staticmethod
    def get_max_simulation_time(num_tasks):
        """根据任务数量估算最大仿真时间"""
        return num_tasks * 200.0