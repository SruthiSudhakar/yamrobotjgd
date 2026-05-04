"""Record/replay trajectories via a running minimum_gello follower server.

Use this when a leader/follower teleop session is already active on the CAN
bus. Instead of opening the bus directly, this connects to the follower's
portal server (default 127.0.0.1:11333) and reads/commands joint positions
through it. Saved trajectories are 7-DoF: 6 arm joints + gripper.
"""

import argparse
import curses
import datetime
import os
import time
from typing import Any

import numpy as np
import portal


class ClientRobot:
    """Portal client for a minimum_gello follower server."""

    def __init__(self, host: str, port: int) -> None:
        self._client = portal.Client(f"{host}:{port}")

    def get_joint_pos(self) -> np.ndarray:
        return self._client.get_joint_pos().result()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self._client.command_joint_pos(joint_pos)

    def move_joints(self, target: np.ndarray, time_interval_s: float = 1.5) -> None:
        start = self.get_joint_pos()
        steps = max(2, int(time_interval_s * 60))
        for i in range(1, steps + 1):
            self.command_joint_pos(start + (target - start) * (i / steps))
            time.sleep(time_interval_s / steps)


def main(stdscr: Any) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-host", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=11333)
    parser.add_argument("--output", type=str, default="./example_trajectory.npy")
    parser.add_argument("--load", type=str, help="Load trajectory from file")
    parser.add_argument("--auto-timestamp", action="store_true",
                        help="Append a UTC timestamp to --output on each save so a new file is written every time.")
    parser.add_argument("--button-server-host", type=str, default="127.0.0.1",
                        help="Host of the leader's button server (run leader with --button_server).")
    parser.add_argument("--button-server-port", type=int, default=0,
                        help="Port of the leader's button server. 0 disables button polling.")
    args, _ = parser.parse_known_args()

    button_client = None
    last_btn1 = 0.0
    if args.button_server_port:
        button_client = portal.Client(f"{args.button_server_host}:{args.button_server_port}")

    robot = ClientRobot(args.server_host, args.server_port)

    # Curses setup
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(0)

    trajectory = []
    timestamps = []
    recording = False
    replaying = False
    replay_idx = 0
    target_freq = 60.0
    dt = 1.0 / target_freq

    if args.load and os.path.exists(args.load):
        try:
            data = np.load(args.load, allow_pickle=True).item()
            trajectory = data["trajectory"].tolist()
            timestamps = data["timestamps"].tolist()
            if "frequency" in data:
                target_freq = data["frequency"]
                dt = 1.0 / target_freq
        except Exception as e:
            print(f"Error loading trajectory: {e}")
            trajectory = []
            timestamps = []

    instructions = [
        "Controls (recording the FOLLOWER as you teleop with the leader):",
        "  r : Start/stop recording",
        "  p : Start replay",
        "  s : Save trajectory",
        "  l : Load trajectory from file",
        "  q : Quit",
        "",
        "Status:",
    ]

    last_record_time = time.monotonic()
    last_replay_time = time.monotonic()

    while True:
        current_time = time.monotonic()

        # Poll leader button[1]: toggle recording on release (1 -> 0).
        if button_client is not None:
            try:
                buttons = button_client.get_button_state().result()
                btn1 = float(buttons[1])
            except Exception:
                btn1 = last_btn1
            if last_btn1 > 0.5 and btn1 < 0.5:
                recording = not recording
                replaying = False
                if recording:
                    trajectory = []
                    timestamps = []
                    last_record_time = current_time
                stdscr.addstr(len(instructions) + 2, 0, f"Recording: {recording} (button)   ")
            last_btn1 = btn1

        key = stdscr.getch()

        if key != -1:
            if key == ord("q"):
                break
            elif key == ord("r"):
                recording = not recording
                replaying = False
                if recording:
                    trajectory = []
                    timestamps = []
                    last_record_time = current_time
                stdscr.addstr(len(instructions) + 2, 0, f"Recording: {recording}         ")
            elif key == ord("p"):
                if len(trajectory) > 0:
                    replaying = True
                    recording = False
                    replay_idx = 0
                    last_replay_time = current_time
                else:
                    stdscr.addstr(len(instructions) + 2, 0, "No trajectory to replay.      ")
            elif key == ord("s"):
                if len(trajectory) > 0:
                    data = {
                        "trajectory": np.array(trajectory),
                        "timestamps": np.array(timestamps),
                        "frequency": target_freq,
                    }
                    if args.auto_timestamp:
                        stem, ext = os.path.splitext(args.output)
                        ext = ext or ".npy"
                        stamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
                        out_path = f"{stem}_{stamp}{ext}"
                    else:
                        out_path = args.output
                    np.save(out_path, data)
                    stdscr.addstr(len(instructions) + 2, 0, f"Saved to {out_path}        ")
                else:
                    stdscr.addstr(len(instructions) + 2, 0, "No trajectory to save.        ")
            elif key == ord("l"):
                stdscr.addstr(len(instructions) + 2, 0, "Enter filename to load: ")
                stdscr.refresh()

                filename = ""
                while True:
                    key = stdscr.getch()
                    if key == ord("\n"):
                        break
                    elif key == ord("\x1b"):
                        filename = ""
                        break
                    elif key == ord("\x7f"):
                        if filename:
                            filename = filename[:-1]
                    elif 32 <= key <= 126:
                        filename += chr(key)

                    stdscr.addstr(len(instructions) + 2, 0, f"Enter filename to load: {filename}")
                    stdscr.refresh()

                if filename and os.path.exists(filename):
                    try:
                        data = np.load(filename, allow_pickle=True).item()
                        trajectory = data["trajectory"].tolist()
                        timestamps = data["timestamps"].tolist()
                        if "frequency" in data:
                            target_freq = data["frequency"]
                            dt = 1.0 / target_freq
                        stdscr.addstr(len(instructions) + 2, 0, f"Loaded {filename} successfully    ")
                    except Exception:
                        stdscr.addstr(len(instructions) + 2, 0, f"Error loading {filename}      ")
                elif filename:
                    stdscr.addstr(len(instructions) + 2, 0, f"File {filename} not found      ")

        stdscr.erase()
        for i, line in enumerate(instructions):
            stdscr.addstr(i, 0, line)
        stdscr.addstr(len(instructions), 0, f"Recording: {recording}  Replaying: {replaying}")
        stdscr.addstr(len(instructions) + 1, 0, f"Trajectory length: {len(trajectory)} samples")
        stdscr.addstr(len(instructions) + 3, 0, "Press 'q' to quit.")

        if recording and (current_time - last_record_time) >= dt:
            qpos = robot.get_joint_pos()
            trajectory.append(np.copy(qpos))
            timestamps.append(current_time)
            last_record_time = current_time

        if replaying and len(trajectory) > 0:
            if replay_idx == 0:
                robot.move_joints(np.array(trajectory[replay_idx]), time_interval_s=1.5)
            if replay_idx < len(trajectory) and (current_time - last_replay_time) >= dt:
                robot.command_joint_pos(trajectory[replay_idx])
                replay_idx += 1
                last_replay_time = current_time
                stdscr.addstr(len(instructions) + 4, 0, f"Replaying: {replay_idx}/{len(trajectory)}")
            elif replay_idx >= len(trajectory):
                replaying = False
                stdscr.addstr(len(instructions) + 4, 0, "Replay finished.              ")

        stdscr.refresh()
        time.sleep(0.02)


if __name__ == "__main__":
    curses.wrapper(main)
