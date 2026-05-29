"""
Calibrate the new leader arm on can6 by matching the canonical zero pose
of the working leader arm on can2.

Workflow:
  1. Kill any minimum_gello / record_replay scripts on can2 and can6.
  2. python calibrate_can6_leader.py
  3. Move can2 by hand until joints read near zero, hold steady, press ENTER.
  4. Physically pose can6 to match can2's pose, press ENTER.
  5. Script runs set_zero on motors 1-6 of can6.
  6. Power-cycle the can6 arm, then run leader mode again.
"""

import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import GripperType

REF = "can2"
TGT = "can6"


def stream_joint_pos(robot, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            pos = robot.get_observations()["joint_pos"][:6]
            line = "  ".join(f"j{i + 1}={p:+.3f}" for i, p in enumerate(pos))
            print(f"\r  {line}    ", end="", flush=True)
        except Exception:
            pass
        time.sleep(0.1)


def main() -> None:
    print(f"Make sure NO scripts are running on {REF} or {TGT}, and both arms are powered.")
    input("Press ENTER to continue...")

    print(f"\nConnecting to reference arm on {REF}...")
    ref = get_yam_robot(channel=REF, gripper_type=GripperType.YAM_TEACHING_HANDLE)
    time.sleep(0.5)

    print(f"\n*** STEP 1: Move {REF} by hand until all 6 joints read near zero. ***")
    print("Streaming joint positions below. Hold the arm at zero, then press ENTER.\n")

    stop = threading.Event()
    t = threading.Thread(target=stream_joint_pos, args=(ref, stop), daemon=True)
    t.start()
    input("")
    stop.set()
    t.join(timeout=1)

    final = ref.get_observations()["joint_pos"][:6]
    print(f"\n\n{REF} final pose (rad): {np.round(final, 4).tolist()}")
    max_dev = float(np.max(np.abs(final)))
    print(f"Max |joint| from zero: {max_dev:.4f} rad")
    if max_dev > 0.1:
        print(f"WARNING: {REF} is not close to zero (>0.1 rad off on at least one joint).")
        if input("Continue anyway? [y/N] ").strip().lower() != "y":
            return

    print(f"\n*** STEP 2: Physically pose {TGT} to match {REF}'s current pose. ***")
    print(f"Look at {REF} and copy its pose joint-by-joint onto {TGT}.")
    input(f"Press ENTER when {TGT} is posed identically to {REF}...")

    print(f"\n*** STEP 3: Saving motor zero offsets on {TGT}... ***")
    set_zero = Path(__file__).resolve().parent / "i2rt" / "motor_config_tool" / "set_zero.py"
    for motor_id in [1, 2, 3, 4, 5, 6]:
        print(f"\n--- motor {motor_id} on {TGT} ---")
        rc = subprocess.run(
            [sys.executable, str(set_zero), "--channel", TGT, "--motor_id", str(motor_id)],
            check=False,
        ).returncode
        if rc != 0:
            print(f"WARNING: motor {motor_id} set_zero exited with code {rc}")
        time.sleep(0.3)

    print(
        f"\nDONE. Power-cycle the {TGT} arm, then verify with:\n"
        f"  python examples/minimum_gello/minimum_gello.py \\\n"
        f"    --gripper yam_teaching_handle --mode leader --can_channel {TGT} \\\n"
        f"    --bilateral_kp 0.05 --button_server\n"
        f"All six joints should now print near 0 in motor_states, and the arm should "
        f"hold position instead of flailing."
    )


if __name__ == "__main__":
    main()
