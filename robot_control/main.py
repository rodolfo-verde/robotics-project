#!/usr/bin/env python
# !/usr/bin/env python3
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import numpy as np
import threading
import time


def open_daemon(use_sim=False, ros2=True):
    import os
    if ros2:
        # for ros2 and me
        os.system("ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=rx150 " + (
            "use_sim:=true" if use_sim else ""))
    else:
        # for ros1 and Jonas
        os.system("roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=rx150 " + (
            "use_sim:=true" if use_sim else ""))


def main():
    threading.Thread(target=open_daemon).start()
    # time.sleep(2)

    bot = InterbotixManipulatorXS(
        robot_model='rx150',
        group_name='arm',
        gripper_name='gripper'
    )

    # make the bot arm to move in a circle

    bot.arm.go_to_home_pose()
    #
    for i in range(0, 360, 20):
        x = 0.2 * np.cos(i * np.pi / 180)
        y = 0.2 * np.sin(i * np.pi / 180)
        z = 0.2
        bot.arm.set_ee_pose_components(x, y, z)

    # make the robot pick up something in front of it (it's on the table)

    bot.arm.go_to_home_pose()

    # bot.arm.set_ee_pose_components(0.2, 0, 0.1)
    # print(bot.gripper.open())
    #
    bot.arm.set_ee_pose_components(0.2, 0, 0.2)
    bot.gripper.grasp()

    bot.arm.set_ee_pose_components(0.2, 0, 0.1)

    bot.gripper.release()

    #
    # bot.arm.go_to_home_pose()

    bot.arm.go_to_sleep_pose()
    bot.shutdown()


if __name__ == '__main__':
    main()
