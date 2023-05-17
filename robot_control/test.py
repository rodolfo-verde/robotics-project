import threading
import time
from controller import Controller
import tkinter as tk


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


def launch_gui(con: Controller) -> None:
    # simple tkinter gui to control the robot
    # it has a stop button, a pause button, a resume button, and close button
    # plus a context menu to run different demos
    root = tk.Tk()
    root.title("Robot Control")
    root.geometry("300x200")

    def stop() -> None:
        con.clear_queue()

    def pause() -> None:
        # gray out itself and enable resume button
        pause_button.config(state=tk.DISABLED)
        resume_button.config(state=tk.NORMAL)
        con.pause()

    def resume() -> None:
        # gray out itself and enable pause button
        pause_button.config(state=tk.NORMAL)
        resume_button.config(state=tk.DISABLED)
        con.resume()

    def close() -> None:
        root.destroy()

    def safety() -> None:
        threading.Thread(target=safety_demo, args=(con,)).start()

    def back_and_forth() -> None:
        for _ in range(10):
            con.go_sleep(wait=False)
            con.go_home(wait=False)

    def pick_and_place() -> None:
        threading.Thread(target=pick_and_place_demo, args=(con,)).start()

    def pick_and_place2() -> None:
        threading.Thread(target=pick_and_place_demo2, args=(con,)).start()

    menu = tk.Menu(root)
    root.config(menu=menu)
    root.protocol("WM_DELETE_WINDOW", close)

    demo_menu = tk.Menu(menu)
    menu.add_cascade(label="Demo", menu=demo_menu)
    demo_menu.add_command(label="Safety", command=safety)
    demo_menu.add_command(label="Back and Forth", command=back_and_forth)
    demo_menu.add_command(label="Pick and Place", command=pick_and_place)
    demo_menu.add_command(label="Pick and Place 2", command=pick_and_place2)

    # add buttons for stop, pause, resume and quit
    stop_button = tk.Button(root, text="Clear Queue", command=stop)
    stop_button.pack()
    pause_button = tk.Button(root, text="Pause", command=pause)
    pause_button.pack()
    resume_button = tk.Button(root, text="Resume", command=resume)
    resume_button.pack()
    close_button = tk.Button(root, text="Close", command=close)
    close_button.pack()
    resume()

    root.mainloop()


def full_demo(con: Controller) -> None:
    pass


def save_join_pos(con: Controller) -> None:
    join_position = list()
    while True:
        x = input("Press enter to print pose, q to quit ")
        if x == "q":
            break
        pos = con.joint_positions
        join_position.append(pos)
        with open("joint_position.txt", "w") as f:
            for item in join_position:
                f.write("%s\n" % str(item))


def main() -> None:
    # open_daemon(use_sim=True, ros2=True)
    controller = Controller(True)
    try:
        # controller.set_xyz_pose(x=.2, y=0, z=.2, roll=0, pitch=0, wait=True)
        launch_gui(controller)
    except Exception as e:
        print(e)
    print("Shutting down")
    controller.shutdown()


if __name__ == '__main__':
    main()
