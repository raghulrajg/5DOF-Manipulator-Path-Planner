"""
lspb_symmetric_planning.py

Adds a SYMMETRIC PLANNING VIEW on top of the existing ServoCalibration,
so you can plan/specify joint targets and limits in a uniform [-90, +90]
range across ALL joints -- including q1 and q5, whose true DH range is
[0, 180] -- WITHOUT changing anything about the real DH kinematics or the
hardware calibration (offsets/servo mapping stay exactly as measured).

The trick: a per-joint constant shift (remap_offset) that is added back
before the angle is used for kinematics or converted to a servo command.
Because it's a pure additive constant, it does not change velocities or
accelerations (qd_plan == qd_true, qdd_plan == qdd_true) -- only the
absolute angle labels shift. LSPB planning works identically either way.

    q_true = q_plan + remap_offset      <- feed q_true to kinematics / IK
    q_plan = q_true - remap_offset      <- what you see when specifying
                                            targets and joint limits

remap_offset_i is computed automatically as the midpoint of joint i's true
DH range (from ServoCalibration), so:
  - q2, q3, q4 (true range already [-90, 90])  -> remap_offset = 0  (no-op)
  - q1, q5     (true range [0, 180])            -> remap_offset = 90
"""

import numpy as np
import time

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
                           calibration gives (e.g. q5 with offset=0 keeps
                           a [0, 180] planning range, not [-90, 90]).
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

# ======================================================================
# Example: your setup, now planned entirely in symmetric -90..+90 space
# ======================================================================
if __name__ == "__main__":
    JOINT_NAMES = ["q1", "q2", "q3", "q4", "q5"]

    # Hardware calibration: UNCHANGED, exactly as measured. This is the
    # ground truth your kinematics and servo commands rely on.
    cal = ServoCalibration(offsets=[0, 90, 90, 90, 0],
                            directions=[1, 1, 1, 1, 1],
                            servo_min=0, servo_max=180)

    view = SymmetricPlanningView(cal)

    print("True DH limits (what kinematics/servo calibration actually use):")
    cal.print_limits(JOINT_NAMES)
    print("\nSymmetric planning-space limits (what you plan/specify targets in):")
    view.print_view(JOINT_NAMES)
    print()

    # ---- Replace with your actual hardware send function ----
    def send_to_robot(servo_angles):
        print(f"servo cmd = {np.round(servo_angles, 1)}")

    # Plan a move directly in the nice symmetric space -- e.g. move q1 and
    # q5 from their symmetric-zero to +45 deg, same as before for q2..q4.
    q0_plan = [0, 0, 0, 0, 0]        # symmetric-space home
    qf_plan = [45, -30, 60, 10, -30] # symmetric-space target (all now -90..90)

    vmax = [60, 60, 60, 90, 90]
    amax = [120, 120, 120, 180, 180]

    # Plan using the PLANNING-space limits (view.q_plan_min/max), not cal's.
    traj = LSPBTrajectory(n_joints=5)
    # Manually attach planning-space limits for the pre-flight check:
    class _LimitCheck:
        q_min, q_max = view.q_plan_min, view.q_plan_max
        def check_within_limits(self, q, names=None):
            bad = [f"{names[i] if names else i}={q[i]:.2f}"
                   for i in range(len(q))
                   if not (self.q_min[i]-1e-6 <= q[i] <= self.q_max[i]+1e-6)]
            if bad:
                raise ValueError("Joint limit violation (planning space): " + "; ".join(bad))
    traj.cal = _LimitCheck()
    traj.joint_names = JOINT_NAMES

    tf = traj.plan(q0_plan, qf_plan, vmax, amax)
    print(f"Synchronized move time: {tf:.3f} s\n")

    CONTROL_RATE_HZ = 100.0
    dt = 1.0 / CONTROL_RATE_HZ
    t0 = time.perf_counter()

    while True:
        loop_start = time.perf_counter()
        t = loop_start - t0

        q_plan, qd, qdd, done = traj.get_state(t)

        # ---- Convert right before kinematics / hardware ----
        q_true = view.to_true(q_plan)     # <- feed q_true to your FK/IK if needed
        servo_cmd = cal.dh_to_servo(q_true)
        send_to_robot(servo_cmd)

        if done:
            print("Move complete.")
            break

        elapsed = time.perf_counter() - loop_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)