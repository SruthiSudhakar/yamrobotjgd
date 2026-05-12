"""Keyboard-teleop planar (x, y) data collection for the YAM arm.

PushT-style 2D action space: WASD moves an (x, y) target in the world plane,
the PlanarController IKs it to 6 arm joints, the gripper is held closed for
the entire session and is NOT recorded. Two RealSense cameras are captured.

Episode lifecycle:
  ENTER : begin a new episode (move to home pose, start recording at recording-Hz)
  SPACE : end the current episode and save (.npy + JPEG sidecar dirs)
  H     : drive to home pose without recording
  Q     : quit

Saved per-episode files (mirrors examples/record_replay_trajectory/...):
  {output_dir}/episode_{NNNNNN}.npy             — dict (see fields below)
  {output_dir}/episode_{NNNNNN}_frames_cam0/    — frame_{i:06d}.jpg (BGR)
  {output_dir}/episode_{NNNNNN}_frames_cam1/

Before launching: position the arm so the gripper is roughly above your
target workspace, EE pointing the way you want it to keep pointing for all
episodes. The script FKs the *current* pose at startup and uses its z and
its rotation matrix as the fixed plane height and EE orientation.

Press CTRL+C in the launching terminal to abort during gripper auto-calibration.
"""

import argparse
import datetime
import logging
import os
import sys
import threading
import time
from typing import List, Optional

import cv2
import numpy as np
import pygame
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.planar_controller import PlanarController
from i2rt.robots.utils import ArmType, GripperType
from i2rt.utils.utils import override_log_level


_NUM_ARM_JOINTS = 6
_GRIPPER_CLOSED_NORMALIZED = 0.0


def _R_from_args(args: argparse.Namespace) -> Optional[np.ndarray]:
    """Build a 3x3 world-frame rotation matrix for the EE from CLI args.

    Precedence: --R-quat > --R-euler-deg > --gripper-down > None (fall back to FK).
    """
    if args.R_quat is not None:
        w, x, y, z = args.R_quat
        # scipy uses (x, y, z, w) order.
        return Rotation.from_quat([x, y, z, w]).as_matrix()
    if args.R_euler_deg is not None:
        roll, pitch, yaw = args.R_euler_deg
        return Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
    if args.gripper_down:
        # Identity rotation of the arm's grasp_site puts its +Z along world +Z,
        # which (verified by FK on the combined arm+LINEAR_4310 model) places
        # the *physical gripper tip* pointing in world -Z (i.e., at the table).
        # Any pure-yaw rotation (0, 0, YAW) also points the gripper down; use
        # --R-euler-deg 0 0 YAW directly to control jaw orientation.
        return Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True).as_matrix()
    return None


def _autoselect_serials(want: int = 2) -> List[str]:
    ctx = rs.context()
    devices = ctx.query_devices()
    serials = [d.get_info(rs.camera_info.serial_number) for d in devices]
    if len(serials) < want:
        raise RuntimeError(f"Need {want} RealSense cameras, found {len(serials)}: {serials}")
    return serials[:want]


def _open_rs_pipeline(serial: str, width: int, height: int, fps: int) -> rs.pipeline:
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(cfg)
    # Discard a few warm-up frames so auto-exposure stabilizes.
    for _ in range(5):
        pipeline.wait_for_frames()
    return pipeline


def _next_episode_index(output_dir: str) -> int:
    if not os.path.isdir(output_dir):
        return 1
    max_idx = 0
    for name in os.listdir(output_dir):
        if name.startswith("episode_") and name.endswith(".npy"):
            stem = name[len("episode_") : -len(".npy")]
            try:
                max_idx = max(max_idx, int(stem))
            except ValueError:
                continue
    return max_idx + 1


class TeleopApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        os.makedirs(args.output_dir, exist_ok=True)

        # --- RealSense ---------------------------------------------------------
        serials = [args.rs_serial_cam0, args.rs_serial_cam1]
        if not all(serials):
            auto = _autoselect_serials(2)
            serials = [s or auto[i] for i, s in enumerate(serials)]
        self.serials = serials
        logging.info(f"Opening RealSense cameras: cam0={serials[0]}, cam1={serials[1]}")
        self.rs_pipes: List[rs.pipeline] = [
            _open_rs_pipeline(serials[0], args.rs_width, args.rs_height, args.rs_fps),
            _open_rs_pipeline(serials[1], args.rs_width, args.rs_height, args.rs_fps),
        ]
        self._latest_color: List[Optional[np.ndarray]] = [None, None]

        # --- Robot -------------------------------------------------------------
        gripper_type = GripperType.from_string_name(args.gripper_type)
        arm_type = ArmType.from_string_name(args.arm_type)
        logging.info(f"Initializing YAM robot on {args.channel} with gripper={gripper_type}")
        self.robot = get_yam_robot(
            channel=args.channel,
            arm_type=arm_type,
            gripper_type=gripper_type,
            zero_gravity_mode=False,
        )
        # First non-zero command after init so we hold position, gripper closed.
        q_now = self.robot.get_joint_pos()
        q_hold = q_now.copy()
        q_hold[_NUM_ARM_JOINTS] = _GRIPPER_CLOSED_NORMALIZED
        self.robot.command_joint_pos(q_hold)
        time.sleep(0.2)

        # --- Planar controller (calibrated from current pose) -----------------
        self.pc = PlanarController(
            z_fixed=args.z,
            workspace_xy=np.array([args.x_range, args.y_range], dtype=np.float64),
        )
        q6_init = self.robot.get_joint_pos()[:_NUM_ARM_JOINTS]
        z_init, R_init, xy_init = self.pc.calibrate_from_current(q6_init)
        logging.info(f"Pose at launch: xy={xy_init}, z={z_init:.4f}")

        # Decide R_fixed. CLI overrides win; otherwise use the FK from launch.
        R_user = _R_from_args(args)
        if R_user is not None:
            self.pc._R_fixed = R_user
            logging.info(f"R_fixed set from CLI:\n{R_user}")
        else:
            logging.info(f"R_fixed = launch-pose FK rotation:\n{R_init}")

        # Decide z_fixed: --lock-current-z keeps launch z; else use --z.
        if not args.lock_current_z:
            self.pc._z_fixed = args.z
        logging.info(f"z_fixed = {self.pc.z_fixed:.4f}")

        # Compute the home pose. If --home-xy was given, IK the user-specified
        # target with the (now-locked) R; otherwise use the current arm pose
        # as home (only sensible if no CLI R override — otherwise we'd jump
        # to a different orientation on the very first H press).
        if args.home_xy is not None:
            home_xy = np.asarray(args.home_xy, dtype=np.float64)
            ok, q_home_solved = self.pc.step(home_xy, q6_init)
            if not ok:
                raise SystemExit(
                    f"IK failed for --home-xy={home_xy.tolist()} with z={self.pc.z_fixed:.4f} "
                    f"and the chosen R. Pick a reachable home, widen --x-range/--y-range, "
                    "use --lock-current-z, or adjust --R-euler-deg / --R-quat / --gripper-down."
                )
            self.q_home = q_home_solved
            self.home_xy = home_xy
            logging.info(f"Home pose set: xy={home_xy.tolist()}, q_home={q_home_solved}")
        else:
            if R_user is not None:
                # IK the launch xy with the new R so q_home matches the locked R.
                ok, q_home_solved = self.pc.step(xy_init, q6_init)
                if not ok:
                    raise SystemExit(
                        "IK failed to solve home pose for the launch xy under the "
                        "CLI-specified R. Pass --home-xy explicitly."
                    )
                self.q_home = q_home_solved
                self.home_xy = xy_init.copy()
                logging.info(f"Home pose = launch xy with CLI R: q_home={q_home_solved}")
            else:
                self.q_home = q6_init.copy()
                self.home_xy = xy_init.copy()
                logging.info("Home pose = current arm pose (no CLI overrides).")

        self.target_xy = self.pc.clip_xy(self.home_xy)
        logging.info(f"Workspace box: x={args.x_range}, y={args.y_range}; speed={args.speed} m/s")

        # --- Pygame ------------------------------------------------------------
        pygame.init()
        pygame.display.set_caption("YAM planar teleop")
        self.screen = pygame.display.set_mode((520, 320))
        self.font = pygame.font.SysFont("monospace", 14)

        # --- Recording state --------------------------------------------------
        self.recording = False
        self.driving = False  # True after first H or ENTER; keeps the arm idle on launch
        self.episode_index = _next_episode_index(args.output_dir)
        self.buf_action: List[np.ndarray] = []
        self.buf_state_xy: List[np.ndarray] = []
        self.buf_joints: List[np.ndarray] = []
        self.buf_ts: List[float] = []
        self.buf_frames_cam0: List[Optional[np.ndarray]] = []
        self.buf_frames_cam1: List[Optional[np.ndarray]] = []

        self.ik_fail_streak = 0
        self.status_msg = "Click this window, then press H to enable driving."

        # --- Timing -----------------------------------------------------------
        self.cmd_hz = float(args.cmd_hz)
        self.rec_hz = float(args.hz)
        self.cmd_dt = 1.0 / self.cmd_hz
        self.rec_dt = 1.0 / self.rec_hz
        self._tick_count = 0
        self._record_every_n_ticks = max(1, int(round(self.cmd_hz / self.rec_hz)))

    # ----------------------------------------------------------------------------
    def _poll_cameras(self) -> None:
        for i, pipe in enumerate(self.rs_pipes):
            fs = pipe.poll_for_frames()
            if fs:
                c = fs.get_color_frame()
                if c:
                    self._latest_color[i] = np.asanyarray(c.get_data())

    def _read_key_vector(self) -> np.ndarray:
        keys = pygame.key.get_pressed()
        dx = float(keys[pygame.K_a]) - float(keys[pygame.K_d])
        dy = float(keys[pygame.K_s]) - float(keys[pygame.K_w])
        v = np.array([dx, dy], dtype=np.float64)
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return v

    def _step_target(self, key_vec: np.ndarray, dt: float) -> None:
        self.target_xy = self.target_xy + self.args.speed * key_vec * dt
        self.target_xy = self.pc.clip_xy(self.target_xy)

    def _command_robot(self) -> Optional[np.ndarray]:
        q_now = self.robot.get_joint_pos()
        q6_now = q_now[:_NUM_ARM_JOINTS]
        ok, q_target6 = self.pc.step(self.target_xy, q6_now)
        if not ok:
            self.ik_fail_streak += 1
            self.status_msg = f"IK failed (streak={self.ik_fail_streak})"
            if self.ik_fail_streak >= 20:
                self._on_ik_giveup()
            return None
        self.ik_fail_streak = 0
        q7 = np.append(q_target6, _GRIPPER_CLOSED_NORMALIZED)
        # schedule_waypoint smooths to the new target; 2*cmd_dt in the future
        # gives the interpolator a slope to follow rather than a step.
        self.robot.schedule_waypoint(q7, time.time() + 2 * self.cmd_dt)
        return q6_now

    def _on_ik_giveup(self) -> None:
        self.status_msg = "Too many IK failures — stopping recording, holding."
        if self.recording:
            self._end_episode(save=False)
        self.robot.clear_waypoints()

    # --- Episode lifecycle -----------------------------------------------------
    def _move_home_blocking(self) -> None:
        """Drive arm to q_home with the gripper held closed (~2 s)."""
        self.robot.clear_waypoints()
        target = np.append(self.q_home, _GRIPPER_CLOSED_NORMALIZED)
        self.robot.move_joints(target, time_interval_s=2.0)
        self.target_xy = self.pc.fk_xy(self.q_home)
        self.target_xy = self.pc.clip_xy(self.target_xy)
        self.driving = True

    def _begin_episode(self) -> None:
        if self.recording:
            self.status_msg = "Already recording — press SPACE to stop & save first."
            return
        self.status_msg = "Moving to home..."
        self._draw()  # render so user sees the status
        self._move_home_blocking()
        self.buf_action.clear()
        self.buf_state_xy.clear()
        self.buf_joints.clear()
        self.buf_ts.clear()
        self.buf_frames_cam0.clear()
        self.buf_frames_cam1.clear()
        self.recording = True
        self.status_msg = f"Recording episode {self.episode_index:06d}..."

    def _end_episode(self, save: bool) -> None:
        if not self.recording:
            self.status_msg = "Not recording — nothing to save."
            return
        self.recording = False
        n = len(self.buf_action)
        if not save or n == 0:
            self.status_msg = f"Discarded episode ({n} samples)."
            return
        self.status_msg = f"Saving episode {self.episode_index:06d} ({n} samples)..."
        self._draw()
        self._save_episode()
        self.status_msg = f"Saved episode {self.episode_index:06d} ({n} samples)."
        self.episode_index += 1

    def _save_episode(self) -> None:
        out_dir = self.args.output_dir
        stem = f"episode_{self.episode_index:06d}"
        out_npy = os.path.join(out_dir, f"{stem}.npy")
        frames_dirs = [os.path.join(out_dir, f"{stem}_frames_cam{i}") for i in range(2)]
        for d in frames_dirs:
            os.makedirs(d, exist_ok=True)

        n = len(self.buf_action)
        image_paths = [np.empty(n, dtype=object), np.empty(n, dtype=object)]
        for i in range(n):
            for cam, buf in enumerate((self.buf_frames_cam0, self.buf_frames_cam1)):
                frame = buf[i]
                rel = os.path.join(f"{stem}_frames_cam{cam}", f"frame_{i:06d}.jpg")
                if frame is not None:
                    cv2.imwrite(os.path.join(out_dir, rel), frame)
                    image_paths[cam][i] = rel
                else:
                    image_paths[cam][i] = ""

        payload = {
            "action": np.asarray(self.buf_action, dtype=np.float32),
            "state_xy": np.asarray(self.buf_state_xy, dtype=np.float32),
            "joint_pos": np.asarray(self.buf_joints, dtype=np.float32),
            "timestamps": np.asarray(self.buf_ts, dtype=np.float64),
            "frequency": float(self.rec_hz),
            "image_paths_cam0": image_paths[0].astype(np.str_),
            "image_paths_cam1": image_paths[1].astype(np.str_),
            "workspace_xy": np.array(
                [self.args.x_range, self.args.y_range], dtype=np.float32
            ),
            "z_fixed": float(self.pc.z_fixed),
            "R_fixed": np.asarray(self.pc.R_fixed, dtype=np.float32),
            "rs_serials": list(self.serials),
            "episode_index": int(self.episode_index),
            "saved_at_utc": datetime.datetime.utcnow().isoformat(),
        }
        np.save(out_npy, payload, allow_pickle=True)

    # --- Drawing ---------------------------------------------------------------
    def _draw(self) -> None:
        self.screen.fill((20, 20, 24))
        lines = [
            f"Recording: {'YES' if self.recording else 'no'}  Episode: {self.episode_index:06d}",
            f"Driving:    {'YES' if self.driving else 'no  (press H or ENTER)'}",
            f"target_xy:  ({self.target_xy[0]:+.4f}, {self.target_xy[1]:+.4f}) m",
            f"workspace:  x={self.args.x_range}, y={self.args.y_range}",
            f"z_fixed:    {self.pc.z_fixed:+.4f} m",
            f"speed:      {self.args.speed} m/s",
            f"buffer:     {len(self.buf_action)} samples",
            "",
            "WASD = move   ENTER = new episode",
            "SPACE = stop & save   H = home (no record)",
            "Q = quit",
            "",
            f"Status: {self.status_msg}",
        ]
        for i, line in enumerate(lines):
            surf = self.font.render(line, True, (220, 220, 230))
            self.screen.blit(surf, (10, 10 + i * 20))
        pygame.display.flip()

    # --- Main loop -------------------------------------------------------------
    def run(self) -> None:
        # Auto-move to the home pose if the user fully specified it.
        # "Fully specified" = both --home-xy and some --R-* CLI override were given;
        # otherwise we'd be guessing where to send the arm and that's not safe.
        if self.args.home_xy is not None and _R_from_args(self.args) is not None:
            logging.info("Auto-moving to home pose on launch...")
            self.status_msg = "Auto-moving to home..."
            self._draw()
            self._move_home_blocking()
            self.status_msg = "At home. Press ENTER to start recording."

        next_deadline = time.monotonic()
        try:
            while True:
                # Drain pygame events; quit + edge-triggered hotkeys.
                quit_requested = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit_requested = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            quit_requested = True
                        elif event.key == pygame.K_RETURN:
                            self._begin_episode()
                            next_deadline = time.monotonic()
                        elif event.key == pygame.K_SPACE:
                            self._end_episode(save=True)
                        elif event.key == pygame.K_h:
                            if self.recording:
                                self.status_msg = "Stop recording first (SPACE)."
                            else:
                                self.status_msg = "Moving to home..."
                                self._draw()
                                self._move_home_blocking()
                                self.status_msg = "At home."
                                next_deadline = time.monotonic()
                if quit_requested:
                    break

                # Camera polling every tick (cheap; non-blocking).
                self._poll_cameras()

                # While not driving (pre-first ENTER/H), the arm just holds the
                # initial command we sent in __init__. We still draw and poll
                # cameras so the UI is responsive.
                q6_now: Optional[np.ndarray] = None
                if self.driving:
                    key_vec = self._read_key_vector()
                    self._step_target(key_vec, self.cmd_dt)
                    q6_now = self._command_robot()

                # Record at rec_hz subset of cmd_hz.
                if self.recording and q6_now is not None and (self._tick_count % self._record_every_n_ticks == 0):
                    state_xy = self.pc.fk_xy(q6_now)
                    self.buf_action.append(self.target_xy.copy())
                    self.buf_state_xy.append(state_xy.astype(np.float64))
                    self.buf_joints.append(q6_now.copy())
                    self.buf_ts.append(time.monotonic())
                    self.buf_frames_cam0.append(
                        None if self._latest_color[0] is None else self._latest_color[0].copy()
                    )
                    self.buf_frames_cam1.append(
                        None if self._latest_color[1] is None else self._latest_color[1].copy()
                    )

                self._draw()
                self._tick_count += 1

                # Deadline scheduling so the loop rate doesn't drift.
                next_deadline += self.cmd_dt
                now = time.monotonic()
                sleep_for = next_deadline - now
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    # We're behind; resync to avoid runaway catchup.
                    next_deadline = now
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        try:
            if self.recording:
                self._end_episode(save=True)
        finally:
            try:
                self.robot.clear_waypoints()
            except Exception:
                pass
            try:
                self.robot.close()
            except Exception:
                pass
            for pipe in self.rs_pipes:
                try:
                    pipe.stop()
                except Exception:
                    pass
            pygame.quit()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--channel", type=str, default="can0")
    p.add_argument("--arm-type", type=str, default="yam")
    p.add_argument("--gripper-type", type=str, default="linear_4310")
    p.add_argument("--output-dir", type=str, default="./expert_data_jgd_PushT")
    p.add_argument("--z", type=float, default=0.10, help="Fixed plane height (m, world frame).")
    p.add_argument(
        "--lock-current-z",
        action="store_true",
        help="Use the current EE z at startup as the plane height (ignores --z).",
    )
    p.add_argument("--x-range", type=float, nargs=2, default=[0.25, 0.55])
    p.add_argument("--y-range", type=float, nargs=2, default=[-0.15, 0.15])
    p.add_argument("--speed", type=float, default=0.15, help="Teleop speed (m/s).")
    p.add_argument(
        "--home-xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="World-frame (x, y) of the home / episode-reset pose. If omitted, home is "
        "wherever the arm physically is when the script launches.",
    )
    p.add_argument(
        "--R-euler-deg",
        dest="R_euler_deg",
        type=float,
        nargs=3,
        default=None,
        metavar=("ROLL", "PITCH", "YAW"),
        help="Fixed EE orientation as world-frame intrinsic-xyz Euler angles in DEGREES "
        "(applied as Rx(roll) then Ry(pitch) then Rz(yaw)). Example: 180 0 0 for gripper-down.",
    )
    p.add_argument(
        "--R-quat",
        dest="R_quat",
        type=float,
        nargs=4,
        default=None,
        metavar=("W", "X", "Y", "Z"),
        help="Fixed EE orientation as a unit quaternion in MuJoCo (w, x, y, z) order. "
        "Overrides --R-euler-deg and --gripper-down.",
    )
    p.add_argument(
        "--gripper-down",
        dest="gripper_down",
        action="store_true",
        help="Shortcut for --R-euler-deg 180 0 0 (180° rotation about world X, EE pointing -Z).",
    )
    p.add_argument("--hz", type=float, default=30.0, help="Recording rate (Hz).")
    p.add_argument("--cmd-hz", type=float, default=60.0, help="Command/IK loop rate (Hz).")
    p.add_argument("--rs-serial-cam0", type=str, default="")
    p.add_argument("--rs-serial-cam1", type=str, default="")
    p.add_argument("--rs-width", type=int, default=1280)
    p.add_argument("--rs-height", type=int, default=720)
    p.add_argument("--rs-fps", type=int, default=30)
    return p.parse_args()


def main() -> None:
    override_log_level(level=logging.INFO)
    args = _parse_args()
    app = TeleopApp(args)
    app.run()


if __name__ == "__main__":
    main()
