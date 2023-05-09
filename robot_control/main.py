#!/usr/bin/env python
from typing import List

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


class RobotController:
    _max_step_size: float = 0.03  # radians (0.02 is the default)
    _moving_time: float = 0.03  # seconds (0.04 is the default)
    _SLEEP_POS = [-0.00920388475060463, -1.7993595600128174, 1.6444274187088013, 0.777728259563446, 0.0]
    _HOME_POS = [0] * 5
    _FINAL_POS_TOLERANCE = 1  # percent

    def __init__(self, initialize: bool = True):
        self.con = InterbotixManipulatorXS(
            robot_model='rx150',
            group_name='arm',
            gripper_name='gripper'
        )
        self._moving: bool = False
        self._pause: bool = False
        self._forcing_stop: bool = False
        self._moving_thread: threading.Thread | None = None
        self.grasp = self.con.gripper.grasp
        self.release = self.con.gripper.release
        self._num_joints = len(self.joint_positions)
        self.con.gripper.set_pressure(1.0)

        # make sure the robot is in the sleep pose in case it was left in the grasp pose from a previous run
        if initialize:
            self._goto_base_pos()

    def _goto_base_pos(self):
        self.go_home()
        self.go_sleep()
        self.release()

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def moving(self) -> bool:
        return self._moving

    @property
    def joint_positions(self) -> list[float]:
        return self.con.arm.get_joint_commands()

    @property
    def translation_rotation_pose(self) -> tuple[float, float, float, float, float]:
        current_pose = self.con.arm.get_ee_pose()
        rotation = current_pose[0:3, 0:3]  # extract the upper 3x3 matrix

        # Compute the roll and pitch angles
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(-rotation[2, 0], np.sqrt(rotation[2, 1] ** 2 + rotation[2, 2] ** 2))
        x, y, z = current_pose[:3, 3]
        return x, y, z, roll, pitch

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
        self._uniformly_move(joint_positions=self._HOME_POS, wait=True)

    def go_sleep(self) -> None:
        self._uniformly_move(joint_positions=self._SLEEP_POS, wait=True)
        # self.arm.go_to_sleep_pose()
        # go a bit lower to make shutdown easier (stop collision)

    def shutdown(self) -> None:
        # force stop any movement
        self.force_stop()
        # set the trajectory time to 2 seconds and the accel time to 0.3 seconds to make the shutdown smoother
        self.con.arm.set_trajectory_time(2.0, .3)
        self._goto_base_pos()
        self.con.shutdown()

    def _get_theta_list_from_pose(self, x: float = None, y: float = None, z: float = None, roll: float = None,
                                  pitch: float = None) -> list[float]:
        return self.con.arm.set_ee_pose_components(x=x, y=y, z=z, roll=roll, pitch=pitch, execute=False)[0]

    def set_xyz_pose(self, *, x: float = None, y: float = None, z: float = None, roll: float = None,
                     pitch: float = None, wait: bool = False, resume: bool = True) -> None:
        # if none values are given, use the current pose
        current_x, current_y, current_z, current_roll, current_pitch = self.translation_rotation_pose
        if x is None:
            x = current_x
        if y is None:
            y = current_y
        if z is None:
            z = current_z
        if roll is None:
            roll = current_roll
        if pitch is None:
            pitch = current_pitch
        self._uniformly_move(x=x, y=y, z=z, roll=roll, pitch=pitch, wait=wait, resume=resume)

    def set_join_pose(self, joint_positions: list[float], *, wait: bool = False, resume: bool = True) -> None:
        self._uniformly_move(joint_positions=joint_positions, wait=wait, resume=resume)

    def _uniformly_move(self, *, joint_positions: list[float] = None, x: float = None, y: float = None, z: float = None,
                        roll: float = None, pitch: float = None, wait: bool = False, resume: bool = True) -> None:
        # check if all values are None
        if joint_positions is None and x is None and y is None and z is None and roll is None and pitch is None:
            raise Exception("Either joint_positions or x, y, z must be given")
        # joint_positions takes priority over x, y, z
        verify_by_joint_positions = joint_positions is not None
        if joint_positions is None:
            joint_positions = self._get_theta_list_from_pose(x=x, y=y, z=z, roll=roll, pitch=pitch)
        # check if the robot is already moving
        if self._moving:
            raise Exception("Robot is already moving")
        # check if the joint_positions list size is correct
        if len(joint_positions) != self.num_joints:
            raise Exception("joint_positions must be a list of 5 floats")
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
                self.con.arm.set_joint_positions(current_joint_positions + step_size * i, moving_time=self._moving_time,
                                                 accel_time=self._moving_time / 2)
            # make sure the robot is at the final position if not printing a warning (but only if i == steps) it can
            # be up to _FINAL_POS_TOLERANCE percent off
            if i == steps:
                # verify the final position is correct if joint_positions was given check if the final position is
                # within _FINAL_POS_TOLERANCE percent of the given joint_positions if x, y, z was given check if the
                # final position is within _FINAL_POS_TOLERANCE percent of the given x, y, z
                if verify_by_joint_positions:
                    # add 2pi to the joint positions to make sure the difference is not zero
                    final_join_pos = np.array(self.joint_positions) + 2 * np.pi
                    joint_pos = np.array(joint_positions) + 2 * np.pi
                    # get the percent off for each joint
                    percent_off = np.abs(final_join_pos - joint_pos) / (2 * np.pi)
                    # print(f"Warning: Robot did not reach final position. "
                    #       f"Final delta max was {max_percent_off * 100:.2f}%\n"
                    #       f"If it is close enough, adjust the _FINAL_POS_TOLERANCE constant in the code.")
                else:
                    # add 1m to the xyz pose to make sure the difference is not zero
                    final_pose = np.array(self.translation_rotation_pose) + 1
                    desired_final_pose = np.array((x, y, z, roll, pitch)) + 1
                    # get the percent off for each joint
                    percent_off = np.abs(final_pose - desired_final_pose)
                    # print(f"Warning: Robot did not reach final position. "
                    #       f"Final delta max was {max_percent_off * 100:.2f}%\n"
                    #       f"If it is close enough, adjust the _FINAL_POS_TOLERANCE constant in the code.")
                max_percent_off, min_percent_off = np.max(percent_off), np.min(percent_off)
                # print(f"The max percent off was {max_percent_off * 100:.2f}%")
                # print(f"The min percent off was {min_percent_off * 100:.2f}%")
                if max_percent_off > self._FINAL_POS_TOLERANCE:
                    print("Warning: Robot did not reach final position.\n"
                          f"Final delta max was {max_percent_off * 100:.2f}%")
            self._moving = False

        self._moving_thread = threading.Thread(target=thread_func, daemon=True)
        self._moving_thread.start()
        if wait:
            self.await_current_movement()

    def pick_up(self, down: float = 0.01) -> None:
        self.release()
        _, _, z, _, _ = self.translation_rotation_pose
        self.set_xyz_pose(z=z - down, wait=True)
        self.grasp()
        self.set_xyz_pose(z=z, wait=True)


def show_safety_demo(con: RobotController) -> None:
    con.set_join_pose([0, 0, 0, 0, 0])
    while con.moving:
        input("Press enter to pause")
        con.pause()
        input("Press enter to resume")
        con.resume()


def back_and_forth(con: RobotController) -> None:
    x, y, z, roll, pitch = (
        0.19113874615010806, 0.013804476166711615, 0.03837648669453313, -0.00306796166114509, 1.5002332143485546)
    ban = True
    for _ in range(100):
        x += 0.005 * (-1 if ban else 1)
        con.set_xyz_pose(x=x, y=y, z=z, roll=roll, pitch=pitch, wait=True)
        if x < 0.12:
            ban = False
        if x > 0.2:
            ban = True


def pick_and_place_demo(con: RobotController) -> None:
    x, y, z, roll, pitch = (
        0.19113874615010806, 0.013804476166711615, 0.03837648669453313, -0.00306796166114509, 1.5002332143485546)
    con.set_xyz_pose(x=x, y=y, z=z + .01, roll=roll, pitch=pitch, wait=True)
    input("Press enter to pick up")
    con.grasp()
    con.set_xyz_pose(x=x + .03, y=y, z=z + .05, roll=roll, pitch=pitch, wait=True)
    time.sleep(1)
    con.set_xyz_pose(x=x + .07, y=y, z=z + .01, roll=roll, pitch=pitch, wait=True)
    con.release()

    con.go_sleep()

    con.set_xyz_pose(x=x + .07, y=y, z=z + .01, roll=roll, pitch=pitch, wait=True)
    con.grasp()
    con.set_xyz_pose(x=x, y=y, z=z + .01, roll=roll, pitch=pitch, wait=True)
    con.release()


def pick_and_place_demo2(con: RobotController) -> None:
    x, y, z, roll, pitch = (
        0.19113874615010806, 0.013804476166711615, 0.03837648669453313, -0.00306796166114509, 1.5002332143485546)
    con.set_xyz_pose(x=x, y=y, z=z + .01, roll=roll, pitch=pitch, wait=True)
    input("Press enter to pick up")
    con.pick_up()
    input()


def main() -> None:
    open_daemon(use_sim=True, ros2=True)
    controller = RobotController(True)
    try:
        pick_and_place_demo2(controller)
        # back_and_forth(controller)
        # show_safety_demo(controller)
        # controller.set_join_pose(
        #     [0.07209710031747818, -0.04601942375302315, 0.4586602747440338, 1.087592363357544, -0.003067961661145091],
        #     wait=True)

        # x, y, z, roll, pitch = (
        #     0.19113874615010806, 0.013804476166711615, 0.03837648669453313, -0.00306796166114509, 1.5002332143485546)
        # controller.set_xyz_pose(x=x, y=y, z=z + .01, roll=roll, pitch=pitch, wait=True)
        # input()
        # controller.grasp()
        # controller.set_xyz_pose(x=x + .03, y=y, z=z + .05, roll=roll, pitch=pitch, wait=True)
        # time.sleep(1)
        # controller.set_xyz_pose(x=x + .07, y=y, z=z + .01, roll=roll, pitch=pitch, wait=True)
        # controller.release()
        #
        # controller.go_sleep()
        #
        # controller.set_xyz_pose(x=x + .07, y=y, z=z + .01, roll=roll, pitch=pitch, wait=True)
        # controller.grasp()
        # controller.set_xyz_pose(x=x, y=y, z=z + .01, roll=roll, pitch=pitch, wait=True)
        # controller.release()

        # controller.pick_up()

        # print(controller.joint_positions)

        # print(controller.arm.get_ee_pose())
        # print(controller.xyz_pose)
        # print(controller.translation_rotation_pose)
        # controller.set_xyz_pose(x=x, y=y, z=z, wait=True)
        # controller.set_join_pose(
        #     [0.07209710031747818, -0.04601942375302315, 0.4586602747440338, 1.6592363357544, -0.003067961661145091],
        #     wait=True)
        # print(controller.xyz_pose)
        # controller.set_xyz_pose(x=-0.1806, y=-0.0099, z=-0.0352, wait=True)
        #
        # controller.grasp()
        #
        # controller.set_xyz_pose(x=0.1806, y=0.0099, z=0.05, wait=True)

        # controller.set_join_pose(
        #     [0.9618059992790222, -0.01840776950120926, 0.5921165943145752, 0.9832817316055298, -0.010737866163253784],
        #     wait=True)

        # controller.set_xyz_pose(x=0.1806, y=0.0099, z=0.0352, wait=True)

        # controller.set_xyz_pose(x=0.0, y=0.0, z=0.0, wait=True)
        # print(controller.xyz_pose)
        # controller.grasp()
        # controller.set_xyz_pose(x=0.2, y=0.2, z=0.2, wait=True)
        # print(controller.xyz_pose)
        # input("Press enter to shutdown")
    except Exception as e:
        print(e)
    finally:
        print("Shutting down")
        controller.shutdown()


if __name__ == '__main__':
    main()
