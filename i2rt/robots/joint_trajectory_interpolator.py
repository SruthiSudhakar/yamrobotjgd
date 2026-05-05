"""Piecewise-linear joint-space trajectory interpolator.

Joint-space analog of
diffusion_policy/diffusion_policy/common/pose_trajectory_interpolator.py.
Drops the SO(3)/Slerp branch — joint angles are treated as a flat R^N vector
and interpolated with scipy.interpolate.interp1d. "Distance" between two
joint vectors is the L-infinity (per-joint max) of their difference, which
matches the same per-joint speed semantics as the existing max_step_rad
clamp in run_diffusion_policy.py.
"""
from typing import Optional, Union
import numbers
import numpy as np
import scipy.interpolate as si


def _joint_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


class JointTrajectoryInterpolator:
    def __init__(self, times: np.ndarray, joints: np.ndarray):
        times = np.asarray(times)
        joints = np.asarray(joints)
        assert len(times) >= 1
        assert len(joints) == len(times)
        if joints.ndim != 2:
            raise ValueError(f"joints must be 2-D (N, dof); got {joints.shape}")

        if len(times) == 1:
            self.single_step = True
            self._times = times
            self._joints = joints
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])
            self.joint_interp = si.interp1d(
                times, joints, axis=0, assume_sorted=True
            )

    @property
    def times(self) -> np.ndarray:
        if self.single_step:
            return self._times
        return self.joint_interp.x

    @property
    def joints(self) -> np.ndarray:
        if self.single_step:
            return self._joints
        return self.joint_interp.y

    def trim(self, start_t: float, end_t: float) -> "JointTrajectoryInterpolator":
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        all_times = np.unique(all_times)
        all_joints = self(all_times)
        return JointTrajectoryInterpolator(times=all_times, joints=all_joints)

    def drive_to_waypoint(
        self,
        joints: np.ndarray,
        time: float,
        curr_time: float,
        max_speed: float = np.inf,
    ) -> "JointTrajectoryInterpolator":
        assert max_speed > 0
        time = max(time, curr_time)

        curr_joints = self(curr_time)
        dist = _joint_distance(curr_joints, joints)
        min_duration = dist / max_speed
        duration = time - curr_time
        duration = max(duration, min_duration)
        assert duration >= 0
        last_waypoint_time = curr_time + duration

        trimmed = self.trim(curr_time, curr_time)
        new_times = np.append(trimmed.times, [last_waypoint_time], axis=0)
        new_joints = np.append(trimmed.joints, np.asarray(joints)[None, :], axis=0)
        return JointTrajectoryInterpolator(times=new_times, joints=new_joints)

    def schedule_waypoint(
        self,
        joints: np.ndarray,
        time: float,
        max_speed: float = np.inf,
        curr_time: Optional[float] = None,
        last_waypoint_time: Optional[float] = None,
    ) -> "JointTrajectoryInterpolator":
        assert max_speed > 0
        if last_waypoint_time is not None:
            assert curr_time is not None

        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                # waypoint is in the past; ignore
                return self
            start_time = max(curr_time, start_time)

            if last_waypoint_time is not None:
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)

        assert start_time <= end_time
        assert end_time <= time
        if last_waypoint_time is not None:
            if time <= last_waypoint_time:
                assert end_time == curr_time
            else:
                assert end_time == max(last_waypoint_time, curr_time)
        if curr_time is not None:
            assert curr_time <= start_time
            assert curr_time <= time

        trimmed = self.trim(start_time, end_time)

        duration = time - end_time
        end_joints = trimmed(end_time)
        dist = _joint_distance(joints, end_joints)
        min_duration = dist / max_speed
        duration = max(duration, min_duration)
        assert duration >= 0
        last_waypoint_time = end_time + duration

        new_times = np.append(trimmed.times, [last_waypoint_time], axis=0)
        new_joints = np.append(trimmed.joints, np.asarray(joints)[None, :], axis=0)
        return JointTrajectoryInterpolator(times=new_times, joints=new_joints)

    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])

        if self.single_step:
            out = np.broadcast_to(self._joints[0], (len(t),) + self._joints[0].shape).copy()
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)
            out = self.joint_interp(t)

        if is_single:
            out = out[0]
        return out
