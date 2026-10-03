import numpy as np

# Reuse the calibration + trajectory classes from before
from lspb_joint_limits import ServoCalibration, LSPBTrajectory


class SymmetricPlanningView:
    def __init__(self, calibration: ServoCalibration, no_remap_joints=None):
        """
        calibration     : ServoCalibration instance (unchanged, true DH<->servo)
        no_remap_joints : list of joint indices (0-based) to EXCLUDE from
                           symmetric re-centering. Those joints keep their
                           true DH zero as-is (remap_offset forced to 0),
                           so their planning range stays whatever the true
                           calibration gives.
        """
        self.cal = calibration
        no_remap_joints = set(no_remap_joints or [])

        auto_offset = (calibration.q_min + calibration.q_max) / 2.0
        self.remap_offset = np.array([
            0.0 if i in no_remap_joints else auto_offset[i]
            for i in range(len(auto_offset))
        ])

        self.q_plan_min = calibration.q_min - self.remap_offset
        self.q_plan_max = calibration.q_max - self.remap_offset

    def to_true(self, q_plan):
        """Planning-space angle -> true DH angle (feed this to kinematics/IK)."""
        return np.asarray(q_plan, dtype=float) + self.remap_offset

    def to_plan(self, q_true):
        """True DH angle -> planning-space angle (for display / target specs)."""
        return np.asarray(q_true, dtype=float) - self.remap_offset

    def print_view(self, joint_names=None):
        n = len(self.remap_offset)
        for i in range(n):
            name = joint_names[i] if joint_names else f"q{i+1}"
            print(f"{name}: plan range [{self.q_plan_min[i]:7.2f}, {self.q_plan_max[i]:7.2f}] "
                  f"deg  (remap_offset={self.remap_offset[i]:+.1f}, "
                  f"true range [{self.cal.q_min[i]:7.2f}, {self.cal.q_max[i]:7.2f}])")