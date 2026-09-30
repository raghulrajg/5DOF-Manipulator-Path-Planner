"""
lspb_quantized_plot.py

Plans an LSPB trajectory from a Cartesian target, using bounded nonlinear
least-squares inverse kinematics (joint limits enforced directly in the
solver), quantizes to 1-degree servo resolution, and plots ideal vs
quantized joint paths. No serial output -- plotting/verification only
(see lspb_ik_serial_send.py for the version that streams to hardware).

================================================================
UPDATED DH TABLE (no qh helper joint)
================================================================
q = [0,0,0,0,0] corresponds directly to hardware home
(servo = [90,180,180,90,0]). Joint limits (degrees):
    theta1: -90 to 90     theta2: -180 to 0     theta3: 0 to 180
    theta4: -90 to 90     theta5: 0 to 180

================================================================
JOINT LIMITS -> SERVO DIRECTION
================================================================
With offsets=[90,180,180,90,0] (unchanged), matching the stated limits
requires directions=[1,1,-1,1,1] -- only q3 needs its servo direction
FLIPPED relative to q1/q2/q4/q5 (verified: [1,1,1,1,1] gives q3 range
[-180,0], not the stated [0,180]; flipping only q3 fixes it, and q2's
[-180,0] already matches with no flip needed -- this is different from
the previous DH table, where q2 also needed flipping).

================================================================
WHY BOUNDED LEAST-SQUARES OPTIMIZATION
================================================================
scipy.optimize.least_squares (trust-region reflective) with bounds=...
enforces joint limits AS PART OF the search and converges reliably even
near joint-limit boundaries, unlike a scalar bounded minimizer or an
unconstrained iterative solver (DLS) needing post-hoc reject/clip.
================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from lspb_joint_limits import ServoCalibration, LSPBTrajectory
from python_arm.kinematics import get_forward_kinematics, inverse_kinematics_optimized

JOINT_NAMES = ["q1", "q2", "q3", "q4", "q5"]
SERVO_RESOLUTION_DEG = 1.0

# ---- Hardware calibration: offsets unchanged, direction per note above ----
cal = ServoCalibration(offsets=[90, 180, 180, 90, 0],
                        directions=[1, 1, -1, 1, 1],   # verify q3 on hardware!
                        servo_min=0, servo_max=180)

# ---- Joint constraints (radians, for the IK optimizer's bounds) ----
JOINT_BOUNDS_DEG = [(-90, 90), (-180, 0), (0, 180), (-90, 90), (0, 180)]
JOINT_BOUNDS_RAD = [(np.radians(lo), np.radians(hi)) for lo, hi in JOINT_BOUNDS_DEG]

print("Joint bounds (deg):")
for i in range(5):
    print(f"  {JOINT_NAMES[i]}: {JOINT_BOUNDS_DEG[i]}")

# ======================================================================
# 1. Solve IK for the Cartesian target
# ======================================================================
q_default = [np.radians(30), np.radians(-26), np.radians(73), np.radians(5), np.radians(0)]
T_target = get_forward_kinematics(q_default)

# Passing q_default as primary_guess biases the solver toward THIS exact
# configuration, rather than an equally-valid but different one -- the
# arm is redundant for some poses (multiple joint sets can reach the
# same pose), so without this the solver has no way to know which one
# you actually meant.
print(T_target)
q_solved, cost, success = inverse_kinematics_optimized(
    T_target, JOINT_BOUNDS_RAD, primary_guess=q_default)

print(f"\nIK cost: {cost:.2e}   success: {success}")
if not success:
    print("WARNING: IK did not converge cleanly -- verify the plot below "
          "carefully, or this target may be unreachable within the given "
          "joint bounds.")

q_true_hw_target = np.degrees(q_solved)   # kin == true-DH-hw directly
print("Solved joint target (deg):", np.round(q_true_hw_target, 2))

# ======================================================================
# 2. Plan the move (true-DH space: 0 = calibrated hardware home)
# ======================================================================
q0_true = [0, 0, 0, 0, 0]
qf_true = q_true_hw_target.tolist()
vmax = [60, 60, 60, 90, 90]
amax = [120, 120, 120, 180, 180]

traj = LSPBTrajectory(n_joints=5)


class _LimitCheck:
    q_min, q_max = cal.q_min, cal.q_max
    def check_within_limits(self, q, names=None):
        bad = [f"{names[i] if names else i}={q[i]:.2f} (allowed "
               f"[{self.q_min[i]:.1f},{self.q_max[i]:.1f}])"
               for i in range(len(q))
               if not (self.q_min[i]-1e-6 <= q[i] <= self.q_max[i]+1e-6)]
        if bad:
            raise ValueError("Joint limit violation: " + "; ".join(bad))


traj.cal = _LimitCheck()
traj.joint_names = JOINT_NAMES

print("\nTrue DH / servo-calibration limits per joint:")
for i in range(5):
    print(f"  {JOINT_NAMES[i]}: [{cal.q_min[i]:.1f}, {cal.q_max[i]:.1f}]")

tf = traj.plan(q0_true, qf_true, vmax, amax)
print(f"\nMove time: {tf:.3f} s")

# ---- Sample the trajectory finely ----
dt = 0.005
t = np.arange(0, tf + dt, dt)

servo_ideal = np.zeros((5, len(t)))
servo_quant = np.zeros((5, len(t)))

for i, ti in enumerate(t):
    q_true = traj.get_state(ti)[0]
    s = cal.dh_to_servo(q_true)
    servo_ideal[:, i] = s
    servo_quant[:, i] = np.round(s / SERVO_RESOLUTION_DEG) * SERVO_RESOLUTION_DEG

# ---- Plot ----
fig, axs = plt.subplots(5, 1, figsize=(9, 13), sharex=True)
for j in range(5):
    axs[j].plot(t, servo_ideal[j], color="tab:blue", linewidth=1.2,
                label="Ideal (continuous LSPB)")
    axs[j].step(t, servo_quant[j], color="tab:red", linewidth=1.0,
                where="post", label=f"Quantized ({SERVO_RESOLUTION_DEG:.0f}° resolution)")
    axs[j].set_ylabel(f"{JOINT_NAMES[j]}\nservo (deg)")
    axs[j].grid(True, alpha=0.3)
axs[0].legend(loc="lower right", fontsize=8)
axs[-1].set_xlabel("Time (s)")
fig.suptitle("IK-optimized LSPB Trajectory: Ideal vs 1° Servo Resolution", y=0.995)
plt.tight_layout()
plt.savefig("lspb_quantized_path.png", dpi=150)
print("Saved plot.")

# ---- Report max quantization error per joint ----
print("\nMax quantization error per joint (deg):")
for j in range(5):
    err = np.max(np.abs(servo_ideal[j] - servo_quant[j]))
    print(f"  {JOINT_NAMES[j]}: {err:.3f} deg")

print("\nQuantized data (q1, q2, q3, q4, q5) at each timestep:")
for i, ti in enumerate(t):
    q_vals = [int(servo_quant[j, i]) for j in range(5)]
    print(f"t = {ti:.3f}s : {q_vals}")