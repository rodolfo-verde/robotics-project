#!/usr/bin/env python
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import numpy as np
import threading
import time


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
    _max_step_size: float = 0.02  # radians (0.02 is the default)
    _moving_time: float = 0.05  # seconds (0.05 is the default)

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
        self.arm.go_to_home_pose()

    def go_sleep(self) -> None:
        self.arm.go_to_sleep_pose()
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

    def uniformly_move(self, joint_positions: list[float], *, wait: bool = False, resume: bool = True) -> None:
        if self._moving:
            raise Exception("Robot is already moving")
        if resume:
            self.resume()

        self._moving = True
        # get the current joint positions
        current_joint_positions = self.arm.get_joint_commands()
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
            self._moving = False

        self._moving_thread = threading.Thread(target=thread_func, daemon=True)
        self._moving_thread.start()
        if wait:
            self.await_current_movement()


def main() -> None:
    open_daemon(use_sim=True, ros2=True)

    controller = RobotController()
    print(controller.joint_positions)

    controller.shutdown()
    # print(controller.arm.get_joint_commands())


if __name__ == '__main__':
    main()
