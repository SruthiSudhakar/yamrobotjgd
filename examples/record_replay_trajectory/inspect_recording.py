"""Quick inspector for a recording produced by record_replay_trajectory_client.py.

Usage:
    python examples/record_replay_trajectory/inspect_recording.py <path/to/run_*.npy>
"""

import os
import sys

import numpy as np


def main(path: str) -> None:
    if not os.path.exists(path):
        sys.exit(f"file not found: {path}")

    data = np.load(path, allow_pickle=True).item()
    base = os.path.dirname(os.path.abspath(path))

    print(f"File: {path}")
    print(f"Top-level keys: {list(data.keys())}")
    print()

    for k, v in data.items():
        if isinstance(v, np.ndarray):
            head = v[:1] if v.size else v
            print(f"[{k}] ndarray  shape={v.shape}  dtype={v.dtype}")
            if v.dtype.kind in ("f", "i", "u"):
                print(f"  min={v.min():.4f}  max={v.max():.4f}  mean={v.mean():.4f}")
            print(f"  sample[0]={head}")
        else:
            print(f"[{k}] {type(v).__name__}: {v}")
        print()

    # Stats specific to this file's expected schema.
    traj = data.get("trajectory")
    ts = data.get("timestamps")
    if traj is not None and ts is not None and len(ts) > 1:
        duration = float(ts[-1] - ts[0])
        n = len(traj)
        print(f"Recording: {n} samples over {duration:.2f} s "
              f"(~{n / duration:.1f} Hz, target {data.get('frequency', '?')} Hz)")
        if traj.ndim == 2:
            print(f"DoF: {traj.shape[1]} (expect 6 arm + 1 gripper = 7)")

    # Image audit.
    paths = data.get("image_paths")
    if paths is not None:
        total = len(paths)
        empty = sum(1 for p in paths if not p)
        present = 0
        sizes = []
        for p in paths:
            if not p:
                continue
            full = os.path.join(base, p)
            if os.path.exists(full):
                present += 1
                sizes.append(os.path.getsize(full))
        print()
        print(f"Images: {total} entries; {empty} empty (no frame at tick); "
              f"{present}/{total - empty} files on disk")
        if sizes:
            kb = np.array(sizes) / 1024
            print(f"  jpg sizes (KB): min={kb.min():.0f}  max={kb.max():.0f}  mean={kb.mean():.0f}  total={kb.sum() / 1024:.1f} MB")
        if total:
            print(f"  first: {paths[0]!r}")
            print(f"  last:  {paths[-1]!r}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <recording.npy>")
    main(sys.argv[1])
