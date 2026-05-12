"""Planar (x, y) end-effector controller for the YAM arm.

Converts a 2D target (x, y) into a 6-vector of arm joint angles via IK, with
z and end-effector orientation held fixed. Designed as the low-level adapter
under a PushT-style 2D action space: the policy outputs (x, y), this class
turns each (x, y) into joints that the existing MotorChainRobot can track.

Uses the arm-only MuJoCo model for IK (6 joints), so the IK output is a
6-vector. The caller is responsible for appending a gripper command (e.g.,
0.0 for "closed") before sending to a 7-DoF MotorChainRobot.
"""

from typing import Optional, Tuple

import numpy as np

from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import ARM_YAM_XML_PATH


class PlanarController:
    def __init__(
        self,
        xml_path: str = ARM_YAM_XML_PATH,
        site_name: str = "grasp_site",
        z_fixed: float = 0.10,
        R_fixed: Optional[np.ndarray] = None,
        workspace_xy: Optional[np.ndarray] = None,
        ik_pos_threshold: float = 1e-3,
        ik_ori_threshold: float = 1e-3,
        ik_max_iters: int = 50,
        ik_dt: float = 0.05,
    ) -> None:
        self._kin = Kinematics(xml_path, site_name)
        self._site_name = site_name
        self._z_fixed = float(z_fixed)
        self._R_fixed = None if R_fixed is None else np.array(R_fixed, dtype=np.float64).copy()
        if workspace_xy is None:
            workspace_xy = np.array([[0.25, 0.55], [-0.15, 0.15]], dtype=np.float64)
        self._ws = np.asarray(workspace_xy, dtype=np.float64)
        assert self._ws.shape == (2, 2)
        assert self._ws[0, 0] < self._ws[0, 1] and self._ws[1, 0] < self._ws[1, 1]
        self._ik_pos_threshold = ik_pos_threshold
        self._ik_ori_threshold = ik_ori_threshold
        self._ik_max_iters = ik_max_iters
        self._ik_dt = ik_dt

    def calibrate_from_current(self, q6: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """FK current arm joints and adopt that pose's z and R as the fixed
        plane height and EE orientation. Returns (z, R, xy) for logging."""
        T = self._kin.fk(np.asarray(q6, dtype=np.float64), self._site_name)
        self._z_fixed = float(T[2, 3])
        self._R_fixed = T[:3, :3].copy()
        xy = T[:2, 3].copy()
        return self._z_fixed, self._R_fixed, xy

    @property
    def z_fixed(self) -> float:
        return self._z_fixed

    @property
    def R_fixed(self) -> np.ndarray:
        if self._R_fixed is None:
            raise RuntimeError(
                "PlanarController.R_fixed is unset; call calibrate_from_current(current_q6) "
                "or pass R_fixed at construction."
            )
        return self._R_fixed

    @property
    def workspace_xy(self) -> np.ndarray:
        return self._ws.copy()

    def clip_xy(self, xy: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(xy, dtype=np.float64), self._ws[:, 0], self._ws[:, 1])

    def step(self, xy: np.ndarray, current_q6: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Solve IK for the target xy with z and R fixed, seeded from current_q6.

        Returns (success, q6). On failure, q6 is the solver's best-effort
        configuration; callers should typically discard it.
        """
        if self._R_fixed is None:
            raise RuntimeError(
                "PlanarController.step called before R_fixed was set; "
                "call calibrate_from_current(current_q6) first."
            )
        xy_clipped = self.clip_xy(xy)
        T = np.eye(4)
        T[:3, :3] = self._R_fixed
        T[:3, 3] = np.array([xy_clipped[0], xy_clipped[1], self._z_fixed])
        return self._kin.ik(
            T,
            self._site_name,
            init_q=np.asarray(current_q6, dtype=np.float64),
            pos_threshold=self._ik_pos_threshold,
            ori_threshold=self._ik_ori_threshold,
            max_iters=self._ik_max_iters,
            dt=self._ik_dt,
        )

    def fk_xy(self, q6: np.ndarray) -> np.ndarray:
        """Forward-kinematics the arm joints to the world (x, y) of the EE site."""
        T = self._kin.fk(np.asarray(q6, dtype=np.float64), self._site_name)
        return T[:2, 3].copy()

    def fk_pose(self, q6: np.ndarray) -> np.ndarray:
        return self._kin.fk(np.asarray(q6, dtype=np.float64), self._site_name)
