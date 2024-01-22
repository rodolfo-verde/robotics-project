import numpy as np

MAX_VEL = .35

PRE_PICK_UP = (0, -1.2655341625213623, 1.1244080066680908, 0.3834952116012573, 0)
PICK_UP_Z = -0.032
DROP_Z = -0.03
BLOCK_HEIGHT = 0.006

WHITE_PICK_UP = [1.56619442, -0.39136674, 0.49749653, 1.46926852, 0]
BLACK_PICK_UP = [-1.56619442, -0.39136674, 0.49749653, 1.46926852, 0]

WHITE_BLACK_PICK_UP = (WHITE_PICK_UP, BLACK_PICK_UP)

A1 = {
    "pos": [-0.4295, -0.1894, 0.6623, 1.055, -0.405],
    "drop_z": -0.007,
    "pick_up_z": -0.008,
}

A2 = {
    "pos": [0.0138, -0.2104, 0.7082, 1.0807, 0.046],
    "drop_z": -0.003,
    "pick_up_z": -0.003,
}

A3 = {
    "pos": [0.4295, -0.1894, 0.6623, 1.055, 0.405],
    "drop_z": -0.007,
    "pick_up_z": -0.008,
}

B1 = {
    "pos": [-0.2884, 0.1089, 0.2148, 1.1965, -0.2823],
    "drop_z": -0.01,
    "pick_up_z": -0.01,
}

B2 = {
    "pos": [0.0169, 0.1032, 0.2044, 1.2539, 0.0506],
    "drop_z": -0.012,
    "pick_up_z": -0.012,
}

B3 = {
    "pos": [0.2884, 0.1089, 0.2148, 1.1965, 0.2823],
    "drop_z": -0.01,
    "pick_up_z": -0.01,
}

C1 = {
    "pos": [-0.2409, 0.4042, -0.3622, 1.4055, -0.1994],
    "drop_z": -0.018,
    "pick_up_z": -0.022,
}

C2 = {
    "pos": [0.0, 0.5419, -0.5774, 1.5969, 0.0506],
    "drop_z": -0.01,
    "pick_up_z": -0.014,
}

C3 = {
    "pos": [0.2409, 0.4042, -0.3622, 1.4055, 0.1994],
    "drop_z": -0.018,
    "pick_up_z": -0.022,
}

BOARD = (
    (A1, A2, A3),
    (B1, B2, B3),
    (C1, C2, C3)
)

HOME = (-0.010737866163253784, -1.8054953813552856, 1.6490293741226196, 0.6151263117790222, -0.00920388475060463)

MAX_STEP_SIZE: float = np.deg2rad(4.5)  # radians Default: 0.5
MOVING_TIME: float = 10 / 100  # seconds Default: 2ms
FINAL_POS_TOLERANCE: float = np.deg2rad(5)  # radians Default: 4

LINEAR_VELOCITY: float = 0.05  # m/s Default: 0.01
MAX_STEP_LINEAR_VELOCITY: float = 0.02  # m/s Default: 0.003

SIMPLE_MOVE = 0
PARTS_MOVE = 1
GRIPPER_MOVE = 2
CARTESIAN_MOVE = 3
PICK_UP = 4
PLACE_DOWN = 5

WHITE = 0
BLACK = 1

WHITE_BLACK = ("X", "O")

A = 0
B = 1
C = 2
