"""
lspb_ik_serial_send.py

Plans an LSPB trajectory to reach a specific Cartesian end-effector pose,
solved via optimization-based inverse kinematics (joint limits enforced
as bounds), then streams the quantized (1-deg resolution) joint
commands over serial at a fixed 40 ms interval, paced in real time.

SIMPLIFIED CALIBRATION: kinematics.py's DH parameters were re-derived so
that q = [0,0,0,0,0] now corresponds directly to hardware home
(servo [90, 180, 180, 90, 0]). There is no more separate "kin space" vs
"true-DH-hw space" -- the DH joint angle IS the calibration angle:

    servo = offset + direction * q_deg
    offsets = [90, 180, 180, 90, 0]

DIRECTION SIGNS ARE NOT YET VERIFIED. directions=[1,1,1,1,1] below is a
placeholder. Before trusting any real move: command a small, safe
positive-q step on each joint alone and confirm it rotates the way
get_forward_kinematics predicts. Flip the sign for any joint that moves
the wrong way, then update `directions` here (and in the quantized-plot
script) to match.

JOINT LIMITS (deg): q1 +/-90, q2 0..180, q3 0..180, q4 +/-90, q5 +/-180
(kinematics.JOINT_LIMITS_DEG). These are enforced directly inside the
IK optimizer (inverse_kinematics_opt) as bounds, and re-checked again
before planning/sending as a second line of defense.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from lspb_joint_limits import ServoCalibration, LSPBTrajectory
from python_arm.kinematics import (
    get_forward_kinematics,
    inverse_kinematics_opt,
    JOINT_LIMITS_DEG,
)

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

cal = ServoCalibration(offsets=[90, 180, 180, 90, 0],
                        directions=[1, 1, 1, 1, 1],   # VERIFY against hardware -- see note above
                        servo_min=0, servo_max=180)

# ======================================================================
# 1. Specify the move: solve IK for the desired Cartesian target
# ======================================================================
T_target = np.array([
    [ 0,     0,     1,    0.4610   ],
    [0.,    -1,    -0,     0],
    [1.,     0.,     0.,     0],
    [ 0.,     0.,     0.,     1.   ]])

q_start_deg = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) 

T_check = get_forward_kinematics(np.deg2rad(np.array([90, 90, 180, 0, 0])), qh=0.0)

q_solved_rad, ik_result = inverse_kinematics_opt(
    T_check, np.deg2rad(q_start_deg), qh=0.0
)
q_target_deg = np.round(np.rad2deg(q_solved_rad))

print("IK optimizer success:", ik_result.success, " final cost:", ik_result.cost)
print("IK solved joint angles (deg):", q_target_deg)

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
print(f"\nMove time: {tf:.3f} s  |  send interval: {SEND_INTERVAL*1000:.0f} ms  "
      f"-> {int(np.ceil(tf/SEND_INTERVAL))+1} samples")


def quantized_servo(t):
    q_deg = traj.get_state(t)[0]
    s = cal.dh_to_servo(q_deg)
    return (np.round(s / SERVO_RESOLUTION_DEG) * SERVO_RESOLUTION_DEG).astype(int)


# ======================================================================
# 3. Plot (ideal vs quantized) for a sanity check before sending
# ======================================================================
dt_plot = 0.005
t_plot = np.arange(0, tf + dt_plot, dt_plot)
servo_ideal = np.zeros((5, len(t_plot)))
servo_quant_plot = np.zeros((5, len(t_plot)))
for i, ti in enumerate(t_plot):
    q_deg = traj.get_state(ti)[0]
    s = cal.dh_to_servo(q_deg)
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
fig.suptitle("IK-target LSPB Trajectory: Ideal vs Quantized", y=0.995)
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