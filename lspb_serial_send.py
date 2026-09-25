"""
lspb_ik_serial_send.py

Full pipeline: Cartesian target (T_target) -> bounded-optimization
Inverse Kinematics -> LSPB trajectory -> 1-deg quantization -> real-time
serial stream to the arm, at a fixed 40 ms interval.

================================================================
SIMPLIFICATION: DH HOME NOW == HARDWARE HOME
================================================================
The DH parameters were rebuilt so that q = [0,0,0,0,0] corresponds
directly to hardware home (servo = [90,180,180,90,0]). The IK solver's
output (in degrees) IS the true-DH-hw value directly -- no shift
conversion needed anymore.

================================================================
JOINT CONSTRAINTS -> SERVO DIRECTION
================================================================
Stated constraints: j1 = +/-90 deg, j2 = 0..180 deg, j4 = +/-90 deg.
With offsets=[90,180,180,90,0], matching these requires
directions=[1,-1,-1,1,1] (q2, q3 flipped relative to q1/q4/q5). q3's
flip is inferred by symmetry with q2 since no constraint was stated for
it -- VERIFY ON HARDWARE. q5's range [0,180] is unconstrained/unchanged.

================================================================
WHY BOUNDED LEAST-SQUARES OPTIMIZATION INSTEAD OF DLS + REJECT/CLIP
================================================================
scipy.optimize.least_squares (trust-region reflective) with bounds=...
enforces joint limits AS PART OF the search -- it cannot step outside
the given range, so there's nothing to reject or clip afterward. It also
converges far more reliably than a scalar bounded minimizer, especially
near joint-limit boundaries. Multi-start (several random starting
points, plus an optional primary_guess) is still used to avoid landing
on a different-but-valid solution branch than the one you intended.
================================================================
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from lspb_joint_limits import ServoCalibration, LSPBTrajectory
from python_arm.kinematics import get_forward_kinematics, inverse_kinematics_optimized

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ======================================================================
# Config
# ======================================================================
JOINT_NAMES = ["q1", "q2", "q3", "q4", "q5"]
SERVO_RESOLUTION_DEG = 1.0
SEND_INTERVAL = 0.040          # 40 ms

SERIAL_ENABLED = True
SERIAL_PORT = "COM20"
SERIAL_BAUD = 115200

# ---- Hardware calibration: offsets unchanged, directions per note above ----
cal = ServoCalibration(offsets=[90, 180, 180, 90, 0],
                        directions=[1, -1, -1, 1, 1],   # verify q2,q3 on hardware!
                        servo_min=0, servo_max=180)

# ---- Joint constraints (for the IK optimizer's bounds) ----
JOINT_BOUNDS_DEG = [(-90, 90), (0, 180), (0, 180), (-90, 90), (0, 180)]
JOINT_BOUNDS_RAD = [(np.radians(lo), np.radians(hi)) for lo, hi in JOINT_BOUNDS_DEG]

print("Joint bounds (deg):")
for i in range(5):
    print(f"  {JOINT_NAMES[i]}: {JOINT_BOUNDS_DEG[i]}")

# ======================================================================
# 1. Solve IK for the Cartesian target
# ======================================================================
q_default = [np.radians(30), np.radians(45), np.radians(90), np.radians(90), np.radians(0)]
qh_fixed = 0.0
T_target = get_forward_kinematics(q_default, qh_fixed)

# Passing q_default as primary_guess biases the solver toward THIS exact
# configuration, rather than an equally-valid but different one -- the
# arm is redundant for some poses (multiple joint sets can reach the
# same pose), so without this the solver has no way to know which one
# you actually meant. If you instead have a raw Cartesian pose (not
# built from known joint angles), set primary_guess=None or supply your
# own preferred configuration in radians.
q_solved, cost, success = inverse_kinematics_optimized(
    T_target, JOINT_BOUNDS_RAD, qh=qh_fixed, primary_guess=q_default)

print(f"\nIK cost: {cost:.2e}   success: {success}")
if not success:
    print("WARNING: IK did not converge cleanly -- verify this target in "
          "lspb_quantized_plot.py's plot BEFORE sending to real hardware. "
          "It may be unreachable within the given joint bounds.")

q_true_hw_target = np.degrees(q_solved)   # kin == true-DH-hw directly now
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
print(f"\nMove time: {tf:.3f} s  |  send interval: {SEND_INTERVAL*1000:.0f} ms  "
      f"-> {int(np.ceil(tf/SEND_INTERVAL))+1} samples")


def quantized_servo(t):
    q_true = traj.get_state(t)[0]
    s = cal.dh_to_servo(q_true)
    return (np.round(s / SERVO_RESOLUTION_DEG) * SERVO_RESOLUTION_DEG).astype(int)


# ======================================================================
# 3. Plot (ideal vs quantized) for a sanity check before sending
# ======================================================================
dt_plot = 0.005
t_plot = np.arange(0, tf + dt_plot, dt_plot)
servo_ideal = np.zeros((5, len(t_plot)))
servo_quant_plot = np.zeros((5, len(t_plot)))
for i, ti in enumerate(t_plot):
    q_true = traj.get_state(ti)[0]
    s = cal.dh_to_servo(q_true)
    servo_ideal[:, i] = s
    servo_quant_plot[:, i] = np.round(s / SERVO_RESOLUTION_DEG) * SERVO_RESOLUTION_DEG

fig, axs = plt.subplots(5, 1, figsize=(9, 13), sharex=True)
for j in range(5):
    axs[j].plot(t_plot, servo_ideal[j], color="tab:blue", linewidth=1.2, label="Ideal")
    axs[j].step(t_plot, servo_quant_plot[j], color="tab:red", linewidth=1.0,
                where="post", label=f"Quantized ({SERVO_RESOLUTION_DEG:.0f}°)")
    axs[j].set_ylabel(f"{JOINT_NAMES[j]}\nservo (deg)")
    axs[j].grid(True, alpha=0.3)
axs[0].legend(loc="lower right", fontsize=8)
axs[-1].set_xlabel("Time (s)")
fig.suptitle("IK-optimized LSPB Trajectory: Ideal vs Quantized", y=0.995)
plt.tight_layout()
plt.savefig("lspb_ik_quantized_path.png", dpi=150)
print("Saved plot.")

# ======================================================================
# 4. Real-time serial send loop @ 40 ms
# ======================================================================
ser = None
if SERIAL_ENABLED and SERIAL_AVAILABLE:
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
        time.sleep(2)
        ser.reset_input_buffer()
        print(f"Serial connected on {SERIAL_PORT} @ {SERIAL_BAUD} baud")
    except Exception as e:
        print(f"Could not open serial port {SERIAL_PORT}: {e}")
        print("Falling back to print-only mode.")
        ser = None
elif SERIAL_ENABLED and not SERIAL_AVAILABLE:
    print("pyserial not installed (pip install pyserial). Print-only mode.")

print("\nStreaming quantized commands...")
n_steps = int(np.ceil(tf / SEND_INTERVAL)) + 1

for step in range(n_steps):
    loop_start = time.perf_counter()
    t_now = min(step * SEND_INTERVAL, tf)

    q_vals = quantized_servo(t_now)
    line = ",".join(str(v) for v in q_vals)

    print(f"t={t_now:.3f}s -> {line}")
    if ser is not None:
        ser.write((line + "\n").encode())

    elapsed = time.perf_counter() - loop_start
    sleep_time = SEND_INTERVAL - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)

if ser is not None:
    ser.close()
    print("Serial connection closed.")

print("Move complete.")