from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from rclpy.logging import LoggingSeverity
import numpy as np
import threading
import time
import queue
from dataclasses import dataclass

_MAX_STEP_SIZE: float = np.deg2rad(0.1)  # radians Default: 0.1
_MOVING_TIME: float = 2 / 1000  # seconds Default: 2ms
_FINAL_POS_TOLERANCE: float = np.deg2rad(0.5)  # radians Default: 0.4
_DEFAULT_SLEEP_POS: list[float] = [0.0, -1.7999999523162842, 1.5499999523162842, 0.800000011920929, 0.0]
_SLEEP_POS: list[float] = [-0.00920388475060463, -1.7993595600128174, 1.6444274187088013, 0.777728259563446, 0.0]
_HOME_POS: list[float] = [0] * 5


class GripperMovement:
    OPEN_AT_START: int = 0
    CLOSE_AT_START: int = 1
    OPEN_AT_END: int = 2
    CLOSE_AT_END: int = 3
    OPEN: int = 4
    CLOSE: int = 5


# class to hold a "movement unit" that will be added to the queue and processed by the moving daemon
# name is not final (maybe MovementUnit?)
@dataclass
class _MovementUnit:
    final_joints: np.ndarray = None
    gripper_movement: int = None


@dataclass
class FinalPose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0


class Controller:
    def __init__(self, initialize: bool = True):
        self._con = InterbotixManipulatorXS(
            robot_model='rx150',
            group_name='arm',
            gripper_name='gripper',
            logging_level=LoggingSeverity.FATAL
        )
        self._moving: bool = False
        self._pause: bool = False
        self._forcing_stop: bool = False
        self._grasp = self._con.gripper.grasp
        self._release = self._con.gripper.release
        self._num_joints = len(self.joint_positions)
        self._con.gripper.set_pressure(1.0)
        self._joint_position_queue = queue.Queue()
        self._con.arm.set_trajectory_time(_MOVING_TIME, _MOVING_TIME / 2)
        self._alive = True

        self._moving_thread = threading.Thread(target=self._moving_daemon, daemon=True)
        self._moving_thread.start()

        # make sure the robot is in the sleep pose in case it was left in the grasp pose from a previous run
        if initialize:
            self._goto_base_pos()

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def queue_size(self) -> int:
        return self._joint_position_queue.qsize()

    @property
    def moving(self) -> bool:
        return self._moving

    @property
    def paused(self) -> bool:
        return self._pause

    @property
    def gripper_moving(self) -> bool:
        return self._con.gripper.gripper_moving

    @property
    def joint_positions(self) -> np.ndarray:
        # return np.array([
        #     self._con.core.joint_states.position[self._con.core.js_index_map[name]]
        #     for name in self._con.arm.group_info.joint_names
        # ])
        return np.array(self._con.arm.get_joint_commands())

    @property
    def translation_rotation_pose(self) -> tuple[float, float, float, float, float]:
        current_pose = self._con.arm.get_ee_pose()
        rotation = current_pose[0:3, 0:3]  # extract the upper 3x3 matrix

        # Compute the roll and pitch angles
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(-rotation[2, 0], np.sqrt(rotation[2, 1] ** 2 + rotation[2, 2] ** 2))
        x, y, z = current_pose[:3, 3]
        return x, y, z, roll, pitch

    def pause(self) -> None:
        self._pause = True

    def resume(self) -> None:
        self._pause = False

    def go_home(self, wait=True) -> None:
        self.add_to_queue(final_joint=_HOME_POS)
        if wait:
            self.await_queue()

    def go_sleep(self, wait=True) -> None:
        self.add_to_queue(final_joint=_DEFAULT_SLEEP_POS)
        self.add_to_queue(final_joint=_SLEEP_POS)
        if wait:
            self.await_queue()

    def shutdown(self) -> None:
        self.resume()
        self.clear_queue()
        self.resume()
        # set the trajectory time to 2 seconds and the accel time to 0.3 seconds to make the shutdown smoother
        # self._con.arm.set_trajectory_time(2.0, .3)
        self._goto_base_pos()
        self._alive = False
        self._moving_thread.join()
        self._con.shutdown()

    def pick_up(self, down: float = 0.01) -> None:
        """Pick up the object in under the gripper"""
        pass

    def add_to_queue(self, *, final_joint: list[float] = None, final_pose: FinalPose = None,
                     gripper_movement: int = None) -> None:
        # check if everything is None and if so raise an error
        if final_joint is None and final_pose is None and gripper_movement is None:
            raise ValueError('At least one of final_joint, final_pose, or gripper must be specified')
        # if final_joint is None, use final pose to set final_joint
        if final_joint is None and final_pose is not None:
            final_joint, _ = self._con.arm.set_ee_pose_components(x=final_pose.x, y=final_pose.y, z=final_pose.z,
                                                                  roll=final_pose.roll, pitch=final_pose.pitch)
        if final_joint is not None:
            final_joint = np.array(final_joint)
        self._joint_position_queue.put(_MovementUnit(final_joint, gripper_movement))

    def await_queue(self) -> None:
        while self._moving_thread.is_alive() and not self._joint_position_queue.empty() or self._moving:
            time.sleep(0.01)

    def clear_queue(self) -> None:
        # pause the moving daemon
        self._forcing_stop = True
        self.resume()
        # wait for the queue to empty
        while not self._joint_position_queue.empty():
            time.sleep(0.01)
        self._forcing_stop = False

    def _check_delta(self, joint_positions: np.ndarray) -> bool:
        return np.all(np.abs(joint_positions - self.joint_positions) < _FINAL_POS_TOLERANCE)

    def _await_gripper(self) -> None:
        while self.gripper_moving:
            time.sleep(0.01)

    def _get_theta_list_from_pose(self, x: float = None, y: float = None, z: float = None, roll: float = None,
                                  pitch: float = None) -> list[float]:
        return self._con.arm.set_ee_pose_components(x=x, y=y, z=z, roll=roll, pitch=pitch, execute=False)[0]

    def _moving_daemon(self) -> None:
        def move_control() -> None:
            for joint_positions in joint_positions_steps:
                while self._pause:
                    if self._forcing_stop:
                        return
                    time.sleep(0.01)
                if self._forcing_stop:
                    return
                self._con.arm.set_joint_positions(joint_positions)
                # verify that the joints moved to the correct position (check if the difference is less than
                # _FINAL_POS_TOLERANCE) if not print a warning and pause the queue
                if not self._check_delta(joint_positions):
                    print('Warning: joint positions not reached, it is recommended to clear the queue. Pausing...')
                    # maybe break here? or clear the queue? or just continue? or raise an error? or do nothing?
                    # decide later rn just pause
                    # self._pause = True
                    # break

        while self._alive:
            while not self._joint_position_queue.empty():
                movement_unit = self._joint_position_queue.get()
                if self._forcing_stop:
                    continue
                while self._pause:
                    if self._forcing_stop:
                        continue
                    time.sleep(0.01)
                desired_joint_positions: np.ndarray = movement_unit.final_joints
                gripper_movement: int = movement_unit.gripper_movement
                if gripper_movement is not None and gripper_movement in [GripperMovement.OPEN, GripperMovement.CLOSE,
                                                                         GripperMovement.OPEN_AT_START,
                                                                         GripperMovement.CLOSE_AT_START]:
                    if gripper_movement == GripperMovement.OPEN:
                        self._release(0)
                    elif gripper_movement == GripperMovement.CLOSE:
                        self._grasp(0)
                    else:
                        if gripper_movement == GripperMovement.OPEN_AT_START:
                            self._release(0)
                        else:
                            self._grasp(0)
                        self._await_gripper()
                if desired_joint_positions is not None:
                    # get the difference between the current and desired joint positions
                    diff = desired_joint_positions - self.joint_positions
                    # check if the absolute difference is less than _FINAL_POS_TOLERANCE and if so do not move (it's
                    # basically already there)
                    if np.all(np.abs(diff) < _FINAL_POS_TOLERANCE):
                        print('Warning: joint positions already reached, skipping...')
                        continue
                    self._moving = True
                    # set the steps to give the step size of _MAX_STEP_SIZE or 1 (whichever is greater)
                    steps = max(1, int(np.max(np.abs(diff)) / _MAX_STEP_SIZE))
                    # get the step size for each joint
                    step_size = diff / steps
                    joint_positions_steps = [self.joint_positions + step_size * (i + 1) for i in range(steps)]
                    joint_positions_steps[-1] = desired_joint_positions
                    move_control()
                if gripper_movement is not None and gripper_movement in [GripperMovement.OPEN_AT_END,
                                                                         GripperMovement.CLOSE_AT_END]:
                    if gripper_movement == GripperMovement.OPEN_AT_END:
                        self._release(0)
                    else:
                        self._grasp(0)
                self._await_gripper()
            self._moving = False
            time.sleep(0.01)

    def _goto_base_pos(self):
        if not self._check_delta(np.array(_SLEEP_POS)):
            self.go_sleep()
        self._release()
