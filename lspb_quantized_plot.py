"""
lspb_ik_quantized_plot.py

Plots ideal (continuous) vs quantized (1-deg servo resolution) joint
paths for an LSPB trajectory whose target pose is solved via
optimization-based inverse kinematics (joint limits enforced as
bounds) -- same planning/quantization logic as lspb_ik_serial_send.py
with the send loop stripped out, for a quick sanity check before
running a move on hardware.

SIMPLIFIED CALIBRATION: kinematics.py's DH parameters were re-derived so
that q = [0,0,0,0,0] now corresponds directly to hardware home
(servo [90, 180, 180, 90, 0]):

    servo = offset + direction * q_deg
    offsets = [90, 180, 180, 90, 0]

DIRECTION SIGNS ARE NOT YET VERIFIED. directions=[1,1,1,1,1] below is a
placeholder -- confirm against hardware (see lspb_ik_serial_send.py's
note) and keep both scripts' `directions` in sync.

JOINT LIMITS (deg): q1 +/-90, q2 0..180, q3 0..180, q4 +/-90, q5 +/-180
(kinematics.JOINT_LIMITS_DEG), enforced directly inside the IK optimizer.
"""

import numpy as np
import matplotlib.pyplot as plt

from lspb_joint_limits import ServoCalibration, LSPBTrajectory
from python_arm.kinematics import (
    get_forward_kinematics,
    inverse_kinematics_opt,
    JOINT_LIMITS_DEG,
)

JOINT_NAMES = ["q1", "q2", "q3", "q4", "q5"]
SERVO_RESOLUTION_DEG = 1.0   # <-- servo accuracy / resolution

cal = ServoCalibration(offsets=[90, 180, 180, 90, 0],
                        directions=[1, 1, 1, 1, 1],   # VERIFY against hardware -- see note above
                        servo_min=0, servo_max=180)

# ======================================================================
# 1. Specify the move: solve IK for the desired Cartesian target
# ======================================================================
T_target = np.array([
    [ 0,     1.,     0.,    -0.   ],
    [-0.,    0.,    -1.,     -0.461],
    [-1.,     0.,     0.,     0.08],
    [ 0.,     0.,     0.,     1.   ]])

q_start_deg = np.array([0.0, 0.0, 0.0, 0.0, 0.0])   # hardware home, directly

q_solved_rad, ik_result = inverse_kinematics_opt(
    T_target, np.deg2rad(q_start_deg), qh=0.0
)
q_target_deg = np.round(np.rad2deg(q_solved_rad))

print("IK optimizer success:", ik_result.success, " final cost:", ik_result.cost)
print("IK solved joint angles (deg):", q_target_deg)

T_check = get_forward_kinematics(q_solved_rad, qh=0.0)
print("Forward-kinematics check (should closely match T_target):")
with np.printoptions(precision=4, suppress=True):
    print(T_check)

# ======================================================================
# 2. Plan the move
# ======================================================================
vmax = [60, 60, 60, 90, 90]
amax = [120, 120, 120, 180, 180]

traj = LSPBTrajectory(n_joints=5)


class _LimitCheck:
    q_min = [lo for lo, hi in JOINT_LIMITS_DEG]
    q_max = [hi for lo, hi in JOINT_LIMITS_DEG]
    def check_within_limits(self, q, names=None):
        bad = [f"{names[i] if names else i}={q[i]:.2f} (allowed "
               f"[{self.q_min[i]:.1f},{self.q_max[i]:.1f}])"
               for i in range(len(q))
               if not (self.q_min[i]-1e-6 <= q[i] <= self.q_max[i]+1e-6)]
        if bad:
            raise ValueError("Joint limit violation: " + "; ".join(bad))


traj.cal = _LimitCheck()
traj.joint_names = JOINT_NAMES

print("\nJoint limits (deg):")
for i in range(5):
    print(f"  {JOINT_NAMES[i]}: [{traj.cal.q_min[i]:.1f}, {traj.cal.q_max[i]:.1f}]")

tf = traj.plan(q_start_deg.tolist(), q_target_deg.tolist(), vmax, amax)
print(f"\nMove time: {tf:.3f} s")

# ---- Sample the trajectory finely ----
dt = 0.005
t = np.arange(0, tf + dt, dt)

servo_ideal = np.zeros((5, len(t)))
servo_quant = np.zeros((5, len(t)))

for i, ti in enumerate(t):
    q_deg = traj.get_state(ti)[0]
    s = cal.dh_to_servo(q_deg)          # ideal continuous servo command
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
fig.suptitle("IK-target LSPB Trajectory: Ideal vs 1° Servo Resolution (all 5 joints)", y=0.995)
plt.tight_layout()
plt.savefig("lspb_ik_quantized_path.png", dpi=150)
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