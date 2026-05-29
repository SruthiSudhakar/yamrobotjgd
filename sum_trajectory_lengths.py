import argparse
import os

import numpy as np


def sum_trajectory_lengths(directory: str) -> int:
    total = 0
    count = 0
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".npy"):
            continue
        path = os.path.join(directory, fname)
        data = np.load(path, allow_pickle=True).tolist()
        length = data["trajectory"].shape[0]
        total += length
        count += 1
        # print(f"{fname}: {length}")
    print(f"\nFiles: {count}")
    print(f"Total trajectory steps: {total}")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str)
    args = parser.parse_args()
    sum_trajectory_lengths(args.directory)
