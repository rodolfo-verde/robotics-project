#!/usr/bin/env python

import threading
import time
from controller import Controller


def open_daemon(*, use_sim: bool = False, ros2: bool = True) -> None:
    import os
    command = "roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=rx150"
    if ros2:
        command = "ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=rx150"
    if use_sim:
        command += " use_sim:=true"
    threading.Thread(target=os.system, args=(command,)).start()


def safety_demo(con: Controller) -> None:
    con.set_join_pose([0, 0, 0, 0, 0])
    while con.moving:
        input("Press enter to pause")
        con.pause()
        input("Press enter to resume")
        con.resume()


def back_and_forth_demo(con: Controller) -> None:
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


def pick_and_place_demo(con: Controller) -> None:
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


def pick_and_place_demo2(con: Controller) -> None:
    x, y, z, roll, pitch = (
        0.19113874615010806, 0.013804476166711615, 0.03837648669453313, -0.00306796166114509, 1.5002332143485546)
    con.set_xyz_pose(x=x, y=y, z=z + .01, roll=roll, pitch=pitch, wait=True)
    input("Press enter to pick up")
    con.pick_up()
    input()


def main() -> None:
    open_daemon(use_sim=True, ros2=True)
    controller = Controller(True)
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
