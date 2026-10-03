import time
import numpy as np
import matplotlib.pyplot as plt

from lspb_joint_limits import ServoCalibration, LSPBTrajectory
from arm5dof import fk, ik, pitch_from_rotation, solve

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

# ---- Hardware calibration: offsets/direction from physical measurement ----
cal = ServoCalibration(offsets=[90, 180, 180, 90, 0],
                        directions=[1, 1, -1, 1, 1],   # verify q3 on hardware!
                        servo_min=0, servo_max=180)

print("Servo-calibration joint limits (deg):")
cal.print_limits(JOINT_NAMES)

# ======================================================================
# 1. Solve IK for the Cartesian target
# ======================================================================
q_default = np.radians([0, -90, 90, 0, 0])
T_target = fk(q_default)
p, R = T_target[:3, 3], T_target[:3, :3]
pitch = pitch_from_rotation(p, R)

# All valid solutions -- diagnostic only, shows how many branches reach
# this pose (the arm can be redundant for some targets).
all_sols = ik(p, pitch, R)
print(f"\nik() found {len(all_sols)} valid solution(s) (deg):")
for s in all_sols:
    print(" ", np.round(np.degrees(s), 2))

q_true_hw_target = np.degrees(all_sols[0])
print("\nSelected joint target (deg):", np.round(q_true_hw_target, 2))

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

print("\nServo-calibration limits per joint (deg), re-checked before planning:")
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
fig.suptitle("IK-solved LSPB Trajectory: Ideal vs Quantized", y=0.995)
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