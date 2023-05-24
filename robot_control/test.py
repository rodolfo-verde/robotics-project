import threading
import time
from controller import Controller, FinalPose, GripperMovement
import tkinter as tk


def open_daemon(*, use_sim: bool = False, ros2: bool = True) -> None:
    import os
    command = "roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=rx150"
    if ros2:
        command = "ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=rx150"
    if use_sim:
        command += " use_sim:=true"
    threading.Thread(target=os.system, args=(command,)).start()


def back_and_forth_demo(con: Controller) -> None:
    x, y, z, roll, pitch = (
        0.19113874615010806, 0.013804476166711615, 0.03837648669453313, -0.00306796166114509, 1.5002332143485546)
    ban = True
    reps = 0
    while reps < 5:
        x += 0.005 * (-1 if ban else 1)
        con.add_to_queue(final_pose=FinalPose(x=x, y=y, z=z, roll=roll, pitch=pitch))
        if x < 0.12:
            ban = False
            con.add_to_queue(gripper_movement=GripperMovement.OPEN)
            reps += 1
        if x > 0.2:
            ban = True
            con.add_to_queue(gripper_movement=GripperMovement.CLOSE)


def pick_and_place_demo(con: Controller) -> None:
    x, y, z, roll, pitch = (
        0.19113874615010806, 0.013804476166711615, 0.03837648669453313, -0.00306796166114509, 1.5002332143485546)
    con.add_to_queue(final_pose=FinalPose(x=x, y=y, z=z + .01, roll=roll, pitch=pitch),
                     gripper_movement=GripperMovement.OPEN)
    con.await_queue()
    input("Press enter to pick up")
    con.add_to_queue(final_pose=FinalPose(x=x + .03, y=y, z=z + .05, roll=roll, pitch=pitch),
                     gripper_movement=GripperMovement.CLOSE_AT_START)
    con.await_queue()
    time.sleep(1)
    con.add_to_queue(final_pose=FinalPose(x=x + .07, y=y, z=z + .01, roll=roll, pitch=pitch),
                     gripper_movement=GripperMovement.OPEN_AT_END)
    con.go_sleep(wait=False)
    con.add_to_queue(final_pose=FinalPose(x=x + .07, y=y, z=z + .01, roll=roll, pitch=pitch),
                     gripper_movement=GripperMovement.CLOSE_AT_END)
    con.add_to_queue(final_pose=FinalPose(x=x, y=y, z=z + .01, roll=roll, pitch=pitch),
                     gripper_movement=GripperMovement.OPEN_AT_END)
    con.go_home(wait=False)


def open_and_close_demo(con: Controller) -> None:
    for _ in range(10):
        con.add_to_queue(gripper_movement=GripperMovement.OPEN)
        con.add_to_queue(gripper_movement=GripperMovement.CLOSE)


def launch_gui(con: Controller) -> None:
    # simple tkinter gui to control the robot
    # it has a stop button, a pause button, a resume button, and close button
    # plus a context menu to run different demos
    root = tk.Tk()
    root.title("Robot Control")
    root.geometry("600x350")

    def stop() -> None:
        con.clear_queue()

    def pause() -> None:
        con.pause()

    def resume() -> None:
        con.resume()

    def close() -> None:
        root.destroy()

    def back_and_forth() -> None:
        threading.Thread(target=back_and_forth_demo, args=(con,)).start()

    def pick_and_place() -> None:
        threading.Thread(target=pick_and_place_demo, args=(con,)).start()

    def open_and_close() -> None:
        threading.Thread(target=open_and_close_demo, args=(con,)).start()

    menu = tk.Menu(root)
    root.config(menu=menu)
    root.protocol("WM_DELETE_WINDOW", close)

    demo_menu = tk.Menu(menu)
    menu.add_cascade(label="Demo", menu=demo_menu)
    demo_menu.add_command(label="Back and Forth", command=back_and_forth)
    demo_menu.add_command(label="Pick and Place", command=pick_and_place)
    demo_menu.add_command(label="Open and Close", command=open_and_close)

    # set up labels for current pose, current joint positions, gripper state, queue_size and update them every 100ms
    pose_label = tk.Label(root, text="Pose: ")
    pose_label.pack()
    joint_label = tk.Label(root, text="Joint Positions: ")
    joint_label.pack()
    gripper_label = tk.Label(root, text="Gripper: ")
    gripper_label.pack()
    queue_size_label = tk.Label(root, text="Queue Size: ")
    queue_size_label.pack()

    def update_label() -> None:
        pose_label.config(text=f"Pose: ({', '.join(str(round(x, 5)) for x in con.translation_rotation_pose)})")
        joint_label.config(text=f"Joint Positions: ({', '.join(str(round(x, 5)) for x in con.joint_positions)})")
        gripper_label.config(text=f"Gripper: {'Moving' if con.gripper_moving else 'Stopped'}")
        queue_size_label.config(text=f"Queue Size: {con.queue_size}")
        # update the button states if paused because the queue size might have changed
        if con.paused:
            pause_button.config(state=tk.DISABLED)
            resume_button.config(state=tk.NORMAL)
        else:
            pause_button.config(state=tk.NORMAL)
            resume_button.config(state=tk.DISABLED)
        root.after(100, update_label)

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

    # add 5 float entry boxes for the pose (x, y, z, roll, pitch) add them inside a frame they are grouped together
    # and add a button to add the pose to the queue
    # pack them 3 and then 2 they are in 2 rows (x, y, z) and (roll, pitch) add a label to the left of each entry
    # box to say what it is
    pose_frame_xyz = tk.Frame(root)
    pose_frame_xyz.pack()
    pose_frame_r_p = tk.Frame(root)
    pose_frame_r_p.pack()
    x_label = tk.Label(pose_frame_xyz, text="X: ")
    x_label.pack(side=tk.LEFT)
    x_entry = tk.Entry(pose_frame_xyz)
    x_entry.pack(side=tk.LEFT)
    y_label = tk.Label(pose_frame_xyz, text="Y: ")
    y_label.pack(side=tk.LEFT)
    y_entry = tk.Entry(pose_frame_xyz)
    y_entry.pack(side=tk.LEFT)
    z_label = tk.Label(pose_frame_xyz, text="Z: ")
    z_label.pack(side=tk.LEFT)
    z_entry = tk.Entry(pose_frame_xyz)
    z_entry.pack(side=tk.LEFT)
    roll_label = tk.Label(pose_frame_r_p, text="Roll: ")
    roll_label.pack(side=tk.LEFT)
    roll_entry = tk.Entry(pose_frame_r_p)
    roll_entry.pack(side=tk.LEFT)
    pitch_label = tk.Label(pose_frame_r_p, text="Pitch: ")
    pitch_label.pack(side=tk.LEFT)
    pitch_entry = tk.Entry(pose_frame_r_p)
    pitch_entry.pack(side=tk.LEFT)

    # add a button to update the entry boxes with the current pose
    def update_pose() -> None:
        # update the Entry boxes with the pose
        x, y, z, roll, pitch = con.translation_rotation_pose
        x_entry.delete(0, tk.END)
        x_entry.insert(0, str(x))
        y_entry.delete(0, tk.END)
        y_entry.insert(0, str(y))
        z_entry.delete(0, tk.END)
        z_entry.insert(0, str(z))
        roll_entry.delete(0, tk.END)
        roll_entry.insert(0, str(roll))
        pitch_entry.delete(0, tk.END)
        pitch_entry.insert(0, str(pitch))

    update_pose_button = tk.Button(root, text="Update Entry with Current Pose", command=update_pose)
    update_pose_button.pack()

    def add_pose() -> None:
        try:
            x = round(float(x_entry.get()), 5)
            y = round(float(y_entry.get()), 5)
            z = round(float(z_entry.get()), 5)
            roll = round(float(roll_entry.get()), 5)
            pitch = round(float(pitch_entry.get()), 5)
            con.add_to_queue(final_pose=FinalPose(x, y, z, roll, pitch))
        except ValueError:
            print("Invalid pose")

    add_pose_button = tk.Button(root, text="Add Pose", command=add_pose)
    add_pose_button.pack()

    update_label()
    root.mainloop()


def full_demo(con: Controller) -> None:
    con = con


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
        launch_gui(controller)
    except Exception as e:
        print(e)
    print("Shutting down")
    controller.shutdown()


if __name__ == '__main__':
    main()
