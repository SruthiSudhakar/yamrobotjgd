"""
Bring up a YAM arm with overridable gravity_comp_factor and ee_mass to find
what value keeps it stable.

Usage:
  # No gravity comp at all — arm should hang limp (support by hand).
  python test_gravity_comp.py --channel can6 --factor 0

  # Try various factors to find one that holds the arm without flailing.
  python test_gravity_comp.py --channel can6 --factor 0.5
  python test_gravity_comp.py --channel can6 --factor 1.0
  python test_gravity_comp.py --channel can6 --factor 1.3     # default for YAM
  python test_gravity_comp.py --channel can6 --factor 1.3 --ee_mass 0.0

For comparison, run the same on the working leader:
  python test_gravity_comp.py --channel can2 --factor 1.3

If factor=0 stops the flailing, gravity comp dynamics are the cause.
Tune from there: increment by 0.1 until the arm holds itself without
oscillating; that's your usable factor for this specific arm.
"""

import argparse
import time

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import GripperType


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="can6")
    parser.add_argument("--gripper", default="yam_teaching_handle")
    parser.add_argument("--factor", type=float, default=0.0, help="Override gravity_comp_factor")
    parser.add_argument("--ee_mass", type=float, default=None, help="Override end-effector mass (kg)")
    args = parser.parse_args()

    gripper_type = GripperType.from_string_name(args.gripper)
    robot = get_yam_robot(channel=args.channel, gripper_type=gripper_type, ee_mass=args.ee_mass)

    print(f"\nOriginal gravity_comp_factor: {robot.gravity_comp_factor}")
    robot.gravity_comp_factor = args.factor
    print(f"Overridden to:                {robot.gravity_comp_factor}")
    print(f"ee_mass override:             {args.ee_mass}")
    print(f"\nObserve the arm. Ctrl+C to exit.\n")

    try:
        while True:
            obs = robot.get_observations()
            pos = obs["joint_pos"][:6]
            eff = obs.get("joint_eff", obs.get("eff", [0.0] * 6))[:6]
            pos_line = "  ".join(f"j{i + 1}={p:+.3f}" for i, p in enumerate(pos))
            eff_line = "  ".join(f"t{i + 1}={e:+.2f}" for i, e in enumerate(eff))
            print(f"\r{pos_line}  |  {eff_line}    ", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    print()


if __name__ == "__main__":
    main()
