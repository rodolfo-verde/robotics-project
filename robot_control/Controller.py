from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from rclpy.logging import LoggingSeverity

import Constants

import os
import math
import time
import numpy as np

# move workdir to robot_control folder
os.chdir(os.path.dirname(os.path.realpath(__file__)))


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
        self._con.arm.set_trajectory_time(Constants.MOVING_TIME, Constants.MOVING_TIME / 2)
        starting_pos = self.get_joint_states()
        diff = final_pos - starting_pos
        if np.all(np.abs(diff) < Constants.FINAL_POS_TOLERANCE):
            print('Warning: joint positions already reached, skipping...')
        # set the steps to give the step size of _MAX_STEP_SIZE or 1 (whichever is greater)
        steps = max(1, int(np.max(np.abs(diff)) / Constants.MAX_STEP_SIZE))
        # get the step size for each joint
        step_size = diff / steps
        joint_positions_steps = [starting_pos + step_size * (i + 1) for i in range(steps)]
        joint_positions_steps[-1] = final_pos
        for pos in joint_positions_steps:
            while self.paused:
                time.sleep(0.5)
            self._con.arm.set_joint_positions(pos)

            if np.all(np.abs(pos - self.get_joint_states()) > Constants.FINAL_POS_TOLERANCE):
                # print the max delta between the desired and actual joint positions and the average delta
                print('Warning: joint positions not reached, skipping...')
                print(
                    f'Max delta: {np.rad2deg(np.max(np.abs(pos - self.get_joint_states())))}, '
                    f'Average delta: {np.rad2deg(np.mean(np.abs(pos - self.get_joint_states())))}')

    def _move_cartesian(self, offset: float) -> None:
        steps = max(abs(math.floor(offset / Constants.MAX_STEP_LINEAR_VELOCITY)), 1)
        mv_t = abs(offset / steps) / Constants.LINEAR_VELOCITY
        self._con.arm.set_trajectory_time(mv_t, mv_t / 2)
        increment = offset / steps
        for i in range(steps):
            while self.paused:
                time.sleep(0.5)
            self._con.arm.set_ee_cartesian_trajectory(z=increment)

    def process_command(self, command: Command):
        if command.code == Constants.SIMPLE_MOVE:
            self._move_joints(command.final_pos)
        elif command.code == Constants.PARTS_MOVE:
            waist_movement = self.get_joint_states()
            waist_movement[0] = command.final_pos[0]
            self._move_joints(waist_movement)
            self._move_joints(command.final_pos)
        elif command.code == Constants.GRIPPER_MOVE:
            self._con.gripper.set_pressure(1.0)
            self._grasp() if command.grasp else self._release()
        elif command.code == Constants.CARTESIAN_MOVE:
            self._move_cartesian(command.z_offset)
        elif command.code == Constants.PICK_UP or command.code == Constants.PLACE_DOWN:
            self._con.gripper.set_pressure(1.0)
            self._move_cartesian(command.z_offset)
            self._grasp() if command.code == Constants.PICK_UP else self._release()
            self._move_cartesian(-command.z_offset * 3)
        else:
            raise ValueError(f"Unknown command code: {command.code}")

    def _check_home(self):
        return np.all(np.abs(self.get_joint_states() - Constants.HOME) < Constants.FINAL_POS_TOLERANCE)

    def goto_home_position(self):
        if not self._check_home():
            self.process_command(Command(code=Constants.SIMPLE_MOVE, final_pos=Constants.PRE_PICK_UP))
            self.process_command(Command(code=Constants.SIMPLE_MOVE, final_pos=Constants.HOME))

    def shutdown(self):
        self.paused = True
        if not self._check_home():
            self._con.arm.set_trajectory_time(3, 1.5)
            self._con.arm.set_joint_positions(Constants.PRE_PICK_UP)
            self._con.arm.set_joint_positions(Constants.HOME)
        self._con.shutdown()


def simple_test():
    controller = Controller()

    controller.process_command(Command(code=Constants.GRIPPER_MOVE, grasp=False))
    controller.process_command(Command(code=Constants.SIMPLE_MOVE, final_pos=Constants.PRE_PICK_UP))
    controller.process_command(Command(code=Constants.SIMPLE_MOVE, final_pos=Constants.WHITE_PICK_UP))

    controller.process_command(Command(code=Constants.CARTESIAN_MOVE, z_offset=Constants.PICK_UP_Z))
    controller.process_command(Command(code=Constants.GRIPPER_MOVE, grasp=True))

    controller.process_command(Command(code=Constants.SIMPLE_MOVE, final_pos=Constants.B2))
    controller.process_command(Command(code=Constants.PLACE_DOWN, z_offset=Constants.B2_Z))

    controller.process_command(Command(code=Constants.SIMPLE_MOVE, final_pos=Constants.PRE_PICK_UP))
    controller.process_command(Command(code=Constants.SIMPLE_MOVE, final_pos=Constants.HOME))
    controller.shutdown()


if __name__ == '__main__':
    simple_test()
