"""
Read-only diagnostic for a YAM arm.

Opens the motor chain on the given CAN channel, sets all commands (torque,
kp, kd, target pos, target vel) to zero, then starts the control thread.
The motors stay enabled and report state continuously, but apply no torque,
so the arm is free to move by hand and the readings update live.

Move the arm by hand and watch the per-joint values. Run a second instance
on can2 in another terminal to compare against the known-good arm.

  python diagnose_can6.py --channel can6
  python diagnose_can6.py --channel can2  # in a second terminal
"""

import argparse
import time

import numpy as np

from i2rt.motor_drivers.dm_driver import DMChainCanInterface, ReceiveMode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="can6")
    args = parser.parse_args()

    motor_list = [
        [0x01, "DM4340"],
        [0x02, "DM4340"],
        [0x03, "DM4340"],
        [0x04, "DM4310"],
        [0x05, "DM4310"],
        [0x06, "DM4310"],
    ]
    n = len(motor_list)

    chain = DMChainCanInterface(
        motor_list,
        [0.0] * n,
        [1] * n,
        args.channel,
        motor_chain_name=f"diag_{args.channel}",
        receive_mode=ReceiveMode.p16,
        start_thread=False,
    )

    # Zero every command parameter before the thread starts sending them.
    z = np.zeros(n)
    chain.set_commands(torques=z, pos=z, vel=z, kp=z, kd=z, get_state=False)
    chain.start_thread()

    print(f"\n[{args.channel}] Motors enabled, zero torque commanded. Move by hand.")
    print(f"[{args.channel}] Ctrl+C to stop.\n")

    try:
        while True:
            states = chain.read_states()
            line = "  ".join(f"j{i + 1}={s.pos:+.3f}" for i, s in enumerate(states))
            print(f"\r[{args.channel}] {line}    ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        chain.close()
        print(f"\n[{args.channel}] Closed.")


if __name__ == "__main__":
    main()
