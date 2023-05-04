#!/usr/bin/env python
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import numpy as np
import threading
import time
import requests

def open_daemon(*, use_sim: bool = False, ros2: bool = True) -> None:
    import os
    command = "roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=rx150"
    if ros2:
        command = "ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=rx150"
    if use_sim:
        command += " use_sim:=true"
    threading.Thread(target=os.system, args=(command,)).start()


def degrees_to_radians(degrees: float) -> float:
    return degrees * np.pi / 180.0


class RobotController(InterbotixManipulatorXS):
    _max_step_size: float = 0.03  # radians (0.02 is the default)
    _moving_time: float = 0.03  # seconds (0.04 is the default)
    _SLEEP_POS = [-0.00920388475060463, -1.7993595600128174, 1.6444274187088013, 0.777728259563446, 0.0]
    _HOME_POS = [0] * 5
    _FINAL_POS_TOLERANCE = 1  # percent

    def __init__(self):
        super().__init__(
            robot_model='rx150',
            group_name='arm',
            gripper_name='gripper'
        )
        self._moving: bool = False
        self._pause: bool = False
        self._forcing_stop: bool = False
        self._moving_thread: threading.Thread | None = None
        self.grasp = self.gripper.grasp
        self.release = self.gripper.release

        # make sure the robot is in the sleep pose in case it was left in the grasp pose from a previous run
        self.go_sleep()
        self.release()

    @property
    def moving(self) -> bool:
        return self._moving

    @property
    def joint_positions(self) -> list[float]:
        return self.arm.get_joint_commands()

    @property
    def xyz_pose(self) -> list[float]:
        return self.arm.get_ee_pose()[:3, 3]

    def await_current_movement(self) -> None:
        if self._moving_thread is not None:
            self._moving_thread.join()

    def force_stop(self) -> None:
        self._forcing_stop = True
        if self._moving_thread is not None:
            self._moving_thread.join()
        self._forcing_stop = False

    def pause(self) -> None:
        self._pause = True

    def resume(self) -> None:
        self._pause = False

    def go_home(self) -> None:
        self.uniformly_move(self._HOME_POS, wait=True)

    def go_sleep(self) -> None:
        self.uniformly_move(self._SLEEP_POS, wait=True)
        # self.arm.go_to_sleep_pose()
        # go a bit lower to make shutdown easier (stop collision)

    def shutdown(self) -> None:
        # force stop any movement
        self.force_stop()
        # set the trajectory time to 2 seconds and the accel time to 0.3 seconds to make the shutdown smoother
        self.arm.set_trajectory_time(2.0, .3)
        # go to sleep pose to make shutdown easier (stop collision)
        self.go_sleep()
        self.release()
        super().shutdown()

    def _get_theta_list_from_pose(self, x: float = None, y: float = None, z: float = None, roll: float = None,
                                  pitch: float = None) -> list[float]:
        # if none values are given, use the current pose
        current_x, current_y, current_z, = self.xyz_pose
        if x is None:
            x = current_x
        if y is None:
            y = current_y
        if z is None:
            z = current_z
        return self.arm.set_ee_pose_components(x=x, y=y, z=z, execute=False)[0]

    def move_to(self, x: float = None, y: float = None, z: float = None, roll: float = None, pitch: float = None,
                *, wait: bool = False, resume: bool = True) -> None:
        if self._moving:
            raise Exception("Robot is already moving")
        if resume:
            self.resume()
        theta_list = self._get_theta_list_from_pose(x=x, y=y, z=z, roll=roll, pitch=pitch)
        print(theta_list)
        self.uniformly_move(theta_list, wait=wait)

    def uniformly_move(self, joint_positions: list[float], *, wait: bool = False, resume: bool = True) -> None:
        if self._moving:
            raise Exception("Robot is already moving")
        if resume:
            self.resume()

        self._moving = True
        current_joint_positions = self.joint_positions
        # get the difference between the current and desired joint positions
        diff = np.array(joint_positions) - np.array(current_joint_positions)
        # set the steps to give the step size of _max_step_size or less and round and make sure it is at least 1
        steps = max(1, int(np.max(np.abs(diff)) / self._max_step_size))
        # get the step size for each joint
        step_size = diff / steps

        def force_stop() -> bool:
            while self._pause:
                if self._forcing_stop:
                    return True
                time.sleep(0.1)
            return False

        # function to run in a thread (to allow for pausing and stopping)
        def thread_func() -> None:
            i = 0
            while not self._forcing_stop and i < steps:
                if force_stop():
                    break
                i += 1
                self.arm.set_joint_positions(current_joint_positions + step_size * i, moving_time=self._moving_time,
                                             accel_time=self._moving_time / 2)
            # make sure the robot is at the final position if not printing a warning (but only if i == steps) it can
            # be up to _FINAL_POS_TOLERANCE percent off
            if i == steps:
                if not np.allclose(self.joint_positions, joint_positions,
                                   rtol=self._FINAL_POS_TOLERANCE / 100.0, atol=0.0):
                    # add 2pi to the joint positions to make sure the difference is not zero
                    final_join_pos = np.array(self.joint_positions) + 2 * np.pi
                    joint_pos = np.array(joint_positions) + 2 * np.pi
                    # get the percent off for each joint
                    percent_off = np.abs(final_join_pos - joint_pos) / (2 * np.pi)
                    # get the max and min percent off
                    max_percent_off = np.max(percent_off)
                    print(f"Warning: Robot did not reach final position. "
                          f"Final delta max was {max_percent_off * 100:.2f}%\n"
                          f"If it is close enough, adjust the _FINAL_POS_TOLERANCE constant in the code.")
            self._moving = False

        self._moving_thread = threading.Thread(target=thread_func, daemon=True)
        self._moving_thread.start()
        if wait:
            self.await_current_movement()


def main() -> None:
    open_daemon(use_sim=False, ros2=True)

    controller = RobotController()
    time.sleep(1)
    try:
        # controller.move_to(z=0.2, wait=True)
        # controller.move_to(z=0.3, wait=True)
        controller.uniformly_move([0, 0, 0, 0, 0, 0])
        while controller.moving:
            input("Press enter to pause")
            controller.pause()
            input("Press enter to resume")
            controller.resume()
        pass
    except Exception as e:
        print(e)
    finally:
        print("Shutting down")
        controller.shutdown()


if __name__ == '__main__':
    main()
