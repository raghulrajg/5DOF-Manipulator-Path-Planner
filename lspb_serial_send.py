"""
lspb_ik_serial_send.py

Full pipeline: Cartesian target (T_target) -> bounded nonlinear
least-squares Inverse Kinematics -> LSPB trajectory -> 1-deg
quantization -> real-time serial stream to the arm, at a fixed 40 ms
interval.

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
FLIPPED relative to q1/q2/q4/q5. VERIFY ON HARDWARE.

================================================================
WHY BOUNDED LEAST-SQUARES OPTIMIZATION
================================================================
scipy.optimize.least_squares (trust-region reflective) with bounds=...
enforces joint limits AS PART OF the search and converges reliably even
near joint-limit boundaries. Multi-start (random starts + an optional
primary_guess) is used to avoid landing on a different-but-valid
solution branch than the one you intended, since the arm can be
redundant for some poses.
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

# ---- Hardware calibration: offsets unchanged, direction per note above ----
cal = ServoCalibration(offsets=[90, 180, 180, 90, 0],
                        directions=[1, 1, -1, 1, 1],   # verify q3 on hardware!
                        servo_min=0, servo_max=180)

# ---- Joint constraints (for the IK optimizer's bounds) ----
JOINT_BOUNDS_DEG = [(-90, 90), (-180, 0), (0, 180), (-90, 90), (0, 180)]
JOINT_BOUNDS_RAD = [(np.radians(lo), np.radians(hi)) for lo, hi in JOINT_BOUNDS_DEG]

print("Joint bounds (deg):")
for i in range(5):
    print(f"  {JOINT_NAMES[i]}: {JOINT_BOUNDS_DEG[i]}")

# ======================================================================
# 1. Solve IK for the Cartesian target
# ======================================================================
q_default = [np.radians(0), np.radians(-90), np.radians(90), np.radians(0), np.radians(0)]
T_target = get_forward_kinematics(q_default)

# Passing q_default as primary_guess biases the solver toward THIS exact
# configuration. If you instead have a raw Cartesian pose (not built
# from known joint angles), set primary_guess=None or supply your own
# preferred configuration in radians.
q_solved, cost, success = inverse_kinematics_optimized(
    T_target, JOINT_BOUNDS_RAD, primary_guess=q_default)

print(f"\nIK cost: {cost:.2e}   success: {success}")
if not success:
    print("WARNING: IK did not converge cleanly -- verify this target in "
          "lspb_quantized_plot.py's plot BEFORE sending to real hardware. "
          "It may be unreachable within the given joint bounds.")

q_true_hw_target = np.degrees(q_solved)
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