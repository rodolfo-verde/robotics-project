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

        starting_pos = self.get_joint_states()
        diff = final_pos - starting_pos
        if np.all(np.abs(diff) < PysicalConstants.FINAL_POS_TOLERANCE):
            # print('Warning: joint positions already reached, skipping...')
            return
        # find the max delta between the desired and actual joint positions
        max_delta = np.max(np.abs(diff))

        mv_t = max_delta / PysicalConstants.MAX_VEL
        self._con.arm.set_trajectory_time(mv_t, mv_t / 2)
        # print(f"Moving time: {mv_t}")

        # self._con.arm.set_trajectory_time(1.7, 1.7 / 2)
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
        self._con.arm.set_trajectory_time(1.0, 1.0 / 2)
        self._con.arm.set_ee_cartesian_trajectory(z=offset)

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
        print(self.get_joint_states())


def new_test(controller: Controller):
    circle = [-0.16260196, 0.01316135, 0.37899045, 1.16483876, -0.1119806]

    controller.process_command(Command(
        code=PysicalConstants.SIMPLE_MOVE,
        final_pos=PysicalConstants.PRE_PICK_UP
    ))

    controller.process_command(Command(
        code=PysicalConstants.SIMPLE_MOVE,
        final_pos=circle
    ))

    controller.process_command(Command(
        code=PysicalConstants.PICK_UP,
        z_offset=-.01 + .005
    ))

    controller.process_command(Command(
        code=PysicalConstants.SIMPLE_MOVE,
        final_pos=PysicalConstants.WHITE_STACK
    ))

    controller.process_command(Command(
        code=PysicalConstants.PLACE_DOWN,
        z_offset=-.01 + .005
    ))

    controller.process_command(Command(
        code=PysicalConstants.SIMPLE_MOVE,
        final_pos=PysicalConstants.PRE_PICK_UP
    ))

    input()

    controller.process_command(Command(
        code=PysicalConstants.SIMPLE_MOVE,
        final_pos=circle
    ))

    controller.process_command(Command(
        code=PysicalConstants.PICK_UP,
        z_offset=-.01
    ))

    controller.process_command(Command(
        code=PysicalConstants.SIMPLE_MOVE,
        final_pos=PysicalConstants.WHITE_STACK
    ))

    controller.process_command(Command(
        code=PysicalConstants.PLACE_DOWN,
        z_offset=-.01
    ))


def simple_test():
    controller = Controller()

    new_test(controller)

    # pi = [0.01227185, 0.21015537, 0.15800002, 1.29621375, -0.00613592]
    #
    # controller.process_command(
    #     Command(
    #         code=PysicalConstants.SIMPLE_MOVE,
    #         final_pos=pi
    #     )
    # )

    # controller._con.arm.set_ee_cartesian_trajectory(z=0.01)
    # controller._con.arm.set_ee_cartesian_trajectory(x=0.005)

    # input()

    # controller.process_command(Command(
    #     code=PysicalConstants.PICK_UP,
    #     z_offset=-.008
    # ))
    #
    # controller.process_command(Command(
    #     code=PysicalConstants.SIMPLE_MOVE,
    #     final_pos=PysicalConstants.PRE_PICK_UP
    # ))
    #
    # controller.process_command(
    #     Command(
    #         code=PysicalConstants.SIMPLE_MOVE,
    #         final_pos=p1
    #     )
    # )
    #
    # controller.process_command(Command(
    #     code=PysicalConstants.PLACE_DOWN,
    #     z_offset=-.008
    # ))

    # c2 = [0.16566993, -0.00460194, 0.53075737, 1.06765068, 0.19174761]
    # c2 = [0.16566993, - 0.03617548, 0.49518711, 1.13479449, 0.19174761]

    # c2 = [0.16566993, 0.03537067, 0.25458371, 1.30385174, 0.19174761]
    #
    # controller.process_command(Command(
    #     code=PysicalConstants.SIMPLE_MOVE,
    #     final_pos=c2
    # ))
    #
    # controller.process_command(Command(
    #     code=PysicalConstants.PICK_UP,
    #     z_offset=-0.022
    # ))
    #
    # controller.process_command(Command(
    #     code=PysicalConstants.SIMPLE_MOVE,
    #     final_pos=c2
    # ))
    #
    # controller.process_command(Command(
    #     code=PysicalConstants.PLACE_DOWN,
    #     z_offset=-0.022
    # ))

    # controller._con.arm.set_ee_cartesian_trajectory(x=-0.02)
    # controller._con.arm.set_ee_cartesian_trajectory(z=0.02)

    controller.print_joint_states()

    # input('Press enter to continue...')

    controller.shutdown()


if __name__ == '__main__':
    simple_test()
