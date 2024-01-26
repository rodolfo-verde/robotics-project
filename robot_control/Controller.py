from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from rclpy.logging import LoggingSeverity

import PysicalConstants
import numpy as np
import time


class Command:
    def __init__(self, code: int, *, final_pos: list | tuple = None, grasp: bool = False, z_offset: float = 0.0):
        self.code = code
        self.final_pos = final_pos
        self.grasp = grasp
        self.z_offset = z_offset


class Controller:
    def __init__(self):
        self._con = InterbotixManipulatorXS(
            robot_model='rx150',
            group_name='arm',
            gripper_name='gripper',
            logging_level=LoggingSeverity.FATAL
        )
        self._paused: bool = False
        self._moving: bool = False

        self._con.gripper.set_pressure(1.0)
        self._grasp = self._con.gripper.grasp
        self._release = self._con.gripper.release
        self.get_joint_states = lambda: np.array(self._con.core.joint_states.position[0:5])

    @property
    def paused(self) -> bool:
        return self._paused

    @paused.setter
    def paused(self, value: bool):
        self._paused = value
        if value:
            time.sleep(0.5)

    def _move_joints(self, final_pos: list | tuple | np.ndarray) -> None:
        # self._con.arm.set_trajectory_time(PysicalConstants.MOVING_TIME, PysicalConstants.MOVING_TIME / 2)
        # starting_pos = self.get_joint_states()
        # diff = final_pos - starting_pos
        # if np.all(np.abs(diff) < PysicalConstants.FINAL_POS_TOLERANCE):
        #     print('Warning: joint positions already reached, skipping...')
        # # set the steps to give the step size of _MAX_STEP_SIZE or 1 (whichever is greater)
        # steps = max(1, int(np.max(np.abs(diff)) / PysicalConstants.MAX_STEP_SIZE))
        # # get the step size for each joint
        # step_size = diff / steps
        # joint_positions_steps = [starting_pos + step_size * (i + 1) for i in range(steps)]
        # joint_positions_steps[-1] = final_pos
        # for pos in joint_positions_steps:
        #     while self.paused:
        #         time.sleep(0.5)
        #     self._con.arm.set_joint_positions(pos)
        #
        #     if np.all(np.abs(pos - self.get_joint_states()) > PysicalConstants.FINAL_POS_TOLERANCE):
        #         # print the max delta between the desired and actual joint positions and the average delta
        #         print('Warning: joint positions not reached, skipping...')
        #         print(
        #             f'Max delta: {np.rad2deg(np.max(np.abs(pos - self.get_joint_states())))}, '
        #             f'Average delta: {np.rad2deg(np.mean(np.abs(pos - self.get_joint_states())))}')

        # starting_pos = self.get_joint_states()
        # diff = final_pos - starting_pos
        # if np.all(np.abs(diff) < PysicalConstants.FINAL_POS_TOLERANCE):
        #     # print('Warning: joint positions already reached, skipping...')
        #     return
        # # find the max delta between the desired and actual joint positions
        # max_delta = np.max(np.abs(diff))
        #
        # mv_t = max_delta / PysicalConstants.MAX_VEL
        # self._con.arm.set_trajectory_time(mv_t, mv_t / 2)
        # # print(f"Moving time: {mv_t}")

        # max_delta = np.max(np.abs(final_pos - self.get_joint_states()))
        # mv_t = max_delta / PysicalConstants.MAX_VEL
        # set the trajectory time
        # self._con.arm.set_trajectory_time(mv_t, mv_t / 2)

        self._con.arm.set_trajectory_time(1.7, 1.7 / 2)
        self._con.arm.set_joint_positions(final_pos)

    def _move_cartesian(self, offset: float) -> None:
        # steps = max(abs(math.floor(offset / PysicalConstants.MAX_STEP_LINEAR_VELOCITY)), 1)
        # mv_t = abs(offset / steps) / PysicalConstants.LINEAR_VELOCITY
        # self._con.arm.set_trajectory_time(mv_t, mv_t / 2)
        # increment = offset / steps
        # for i in range(steps):
        #     while self.paused:
        #         time.sleep(0.5)
        #     self._con.arm.set_ee_cartesian_trajectory(z=increment)
        self._con.arm.set_trajectory_time(0.8, 0.8 / 2)
        self._con.arm.set_ee_cartesian_trajectory(z=offset)

        # mv_t = abs(offset) / PysicalConstants.LINEAR_VELOCITY
        # self._con.arm.set_trajectory_time(mv_t, mv_t / 2)
        # self._con.arm.set_ee_cartesian_trajectory(z=offset)

    def process_command(self, command: Command):
        if command.code == PysicalConstants.SIMPLE_MOVE:
            self._move_joints(command.final_pos)
        elif command.code == PysicalConstants.PARTS_MOVE:
            waist_movement = self.get_joint_states()
            waist_movement[0] = command.final_pos[0]
            self._move_joints(waist_movement)
            self._move_joints(command.final_pos)
        elif command.code == PysicalConstants.GRIPPER_MOVE:
            self._con.gripper.set_pressure(1.0)
            self._grasp() if command.grasp else self._release()
        elif command.code == PysicalConstants.CARTESIAN_MOVE:
            self._move_cartesian(command.z_offset)
        elif command.code == PysicalConstants.PICK_UP or command.code == PysicalConstants.PLACE_DOWN:
            self._con.gripper.set_pressure(1.0)
            self._move_cartesian(command.z_offset)
            self._grasp() if command.code == PysicalConstants.PICK_UP else self._release()
            self._move_cartesian(-command.z_offset * 3)
        else:
            raise ValueError(f"Unknown command code: {command.code}")

    def _check_home(self):
        return np.all(np.abs(self.get_joint_states() - PysicalConstants.HOME) < PysicalConstants.FINAL_POS_TOLERANCE)

    def goto_home_position(self):
        if not self._check_home():
            self.process_command(Command(code=PysicalConstants.SIMPLE_MOVE, final_pos=PysicalConstants.PRE_PICK_UP))
            self.process_command(Command(code=PysicalConstants.SIMPLE_MOVE, final_pos=PysicalConstants.HOME))

    def shutdown(self):
        self.paused = True
        if not self._check_home():
            self._con.arm.set_trajectory_time(3, 1.5)
            self._con.arm.set_joint_positions(PysicalConstants.PRE_PICK_UP)
            self._con.arm.set_joint_positions(PysicalConstants.HOME)
            self._release()
        self._con.shutdown()

    def print_joint_states(self):
        print(f"[{', '.join([str(round(x, 4)) for x in self.get_joint_states()])}]")
