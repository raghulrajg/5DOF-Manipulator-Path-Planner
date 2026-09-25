import tkinter as tk
from tkinter import ttk
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ======================================================================
# Core math (same as the standalone scripts, kept self-contained here)
# ======================================================================
class ServoCalibration:
    def __init__(self, offsets, directions, servo_min=0.0, servo_max=180.0):
        self.offsets = np.asarray(offsets, dtype=float)
        self.n = len(offsets)
        self.directions = np.asarray(directions, dtype=float)
        self.servo_min = servo_min
        self.servo_max = servo_max

        self.q_min = np.zeros(self.n)
        self.q_max = np.zeros(self.n)
        for i in range(self.n):
            d, o = self.directions[i], self.offsets[i]
            a = (servo_min - o) / d
            b = (servo_max - o) / d
            self.q_min[i] = min(a, b)
            self.q_max[i] = max(a, b)

    def dh_to_servo(self, q_dh, clamp=True):
        q_dh = np.asarray(q_dh, dtype=float)
        servo = self.directions * q_dh + self.offsets
        if clamp:
            servo = np.clip(servo, self.servo_min, self.servo_max)
        return servo


class SymmetricPlanningView:
    def __init__(self, calibration, no_remap_joints=None):
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
        return np.asarray(q_plan, dtype=float) + self.remap_offset


def _min_time_single_joint(h, vmax, amax):
    h = abs(h)
    if h == 0:
        return 0.0, 0.0
    vmax, amax = abs(vmax), abs(amax)
    t_to_vmax = vmax / amax
    dist_accel_decel = amax * t_to_vmax ** 2
    if dist_accel_decel >= h:
        tb = np.sqrt(h / amax)
        tf = 2 * tb
    else:
        tb = t_to_vmax
        tf = 2 * tb + (h - dist_accel_decel) / vmax
    return tf, tb


class LSPBTrajectory:
    def __init__(self, n_joints):
        self.n = n_joints
        self.q0 = self.qf = self.tb = self.a = self.v = self.tf = None

    def plan(self, q0, qf, vmax, amax, min_tf=0.05):
        q0 = np.asarray(q0, dtype=float)
        qf = np.asarray(qf, dtype=float)
        vmax = np.asarray(vmax, dtype=float)
        amax = np.asarray(amax, dtype=float)

        joint_min_tf = np.array([
            _min_time_single_joint(qf[j] - q0[j], vmax[j], amax[j])[0]
            for j in range(self.n)
        ])
        tf = max(joint_min_tf.max(), min_tf)

        tb = np.zeros(self.n); a = np.zeros(self.n); v = np.zeros(self.n)
        for j in range(self.n):
            h = qf[j] - q0[j]
            if h == 0:
                tb[j] = tf / 4
                continue
            aj = amax[j] * np.sign(h)
            disc = aj**2 * tf**2 - 4 * aj * h
            tb[j] = tf/2 if disc < 0 else np.clip(tf/2 - np.sqrt(disc)/(2*aj), 1e-6, tf/2)
            a[j] = h / (tb[j] * (tf - tb[j]))
            v[j] = a[j] * tb[j]

        self.q0, self.qf, self.tb, self.a, self.v, self.tf = q0, qf, tb, a, v, tf
        return tf

    def get_state(self, t):
        q = np.zeros(self.n)
        ti = min(max(t, 0.0), self.tf)
        for j in range(self.n):
            q0, qf, tb, a, v, tf = (self.q0[j], self.qf[j], self.tb[j],
                                     self.a[j], self.v[j], self.tf)
            if qf == q0:
                q[j] = q0; continue
            if ti <= tb:
                q[j] = q0 + 0.5*a*ti**2
            elif ti <= tf - tb:
                q[j] = q0 + a*tb*(ti - tb/2)
            else:
                q[j] = qf - 0.5*a*(tf-ti)**2
        return q


# ======================================================================
# GUI
# ======================================================================
JOINT_NAMES = ["q1", "q2", "q3", "q4", "q5"]
DEFAULTS = dict(
    offset=[0, 90, 90, 90, 0],
    direction=[1, 1, 1, 1, 1],
    no_remap=[False, False, False, False, True],
    q0=[0, 0, 0, 0, 0],
    qf=[45, -30, 60, 10, 60],
    vmax=[60, 60, 60, 90, 90],
    amax=[120, 120, 120, 180, 180],
)


class LSPBApp:
    def __init__(self, root):
        self.root = root
        root.title("LSPB Trajectory Planner")

        self.vars = {k: [] for k in
                     ["offset", "direction", "no_remap", "q0", "qf", "vmax", "amax"]}

        table = ttk.Frame(root, padding=10)
        table.grid(row=0, column=0, sticky="w")

        headers = ["Joint", "Offset", "Dir", "No-remap", "q0", "qf", "vmax", "amax"]
        for c, h in enumerate(headers):
            ttk.Label(table, text=h, font=("", 9, "bold")).grid(row=0, column=c, padx=4)

        for i in range(5):
            ttk.Label(table, text=JOINT_NAMES[i]).grid(row=i+1, column=0)

            e_off = tk.DoubleVar(value=DEFAULTS["offset"][i])
            ttk.Entry(table, textvariable=e_off, width=6).grid(row=i+1, column=1)
            self.vars["offset"].append(e_off)

            e_dir = tk.StringVar(value=str(DEFAULTS["direction"][i]))
            ttk.Combobox(table, textvariable=e_dir, values=["1", "-1"],
                         width=4, state="readonly").grid(row=i+1, column=2)
            self.vars["direction"].append(e_dir)

            e_nr = tk.BooleanVar(value=DEFAULTS["no_remap"][i])
            ttk.Checkbutton(table, variable=e_nr).grid(row=i+1, column=3)
            self.vars["no_remap"].append(e_nr)

            for key, col in [("q0", 4), ("qf", 5), ("vmax", 6), ("amax", 7)]:
                v = tk.DoubleVar(value=DEFAULTS[key][i])
                ttk.Entry(table, textvariable=v, width=6).grid(row=i+1, column=col)
                self.vars[key].append(v)

        # Global params
        gframe = ttk.Frame(root, padding=(10, 0))
        gframe.grid(row=1, column=0, sticky="w")
        self.servo_min = tk.DoubleVar(value=0)
        self.servo_max = tk.DoubleVar(value=180)
        self.resolution = tk.DoubleVar(value=1.0)
        for label, var in [("Servo min", self.servo_min),
                            ("Servo max", self.servo_max),
                            ("Resolution (deg)", self.resolution)]:
            ttk.Label(gframe, text=label).pack(side="left", padx=(0, 4))
            ttk.Entry(gframe, textvariable=var, width=6).pack(side="left", padx=(0, 12))

        ttk.Button(gframe, text="Plan & Plot", command=self.plan_and_plot).pack(side="left")

        # Status text
        self.status = tk.Text(root, height=8, width=90, font=("Courier", 9))
        self.status.grid(row=2, column=0, padx=10, pady=(6, 6), sticky="w")

        # Matplotlib figure
        self.fig = Figure(figsize=(9, 9))
        self.axs = self.fig.subplots(5, 1, sharex=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=3, column=0, padx=10, pady=(0, 10))

        self.plan_and_plot()

    def _read(self):
        offset = [v.get() for v in self.vars["offset"]]
        direction = [float(v.get()) for v in self.vars["direction"]]
        no_remap = [i for i, v in enumerate(self.vars["no_remap"]) if v.get()]
        q0 = [v.get() for v in self.vars["q0"]]
        qf = [v.get() for v in self.vars["qf"]]
        vmax = [v.get() for v in self.vars["vmax"]]
        amax = [v.get() for v in self.vars["amax"]]
        return offset, direction, no_remap, q0, qf, vmax, amax

    def plan_and_plot(self):
        self.status.delete("1.0", tk.END)
        try:
            offset, direction, no_remap, q0, qf, vmax, amax = self._read()

            cal = ServoCalibration(offset, direction,
                                    self.servo_min.get(), self.servo_max.get())
            view = SymmetricPlanningView(cal, no_remap_joints=no_remap)

            lines = []
            bad = False
            for i in range(5):
                lines.append(f"{JOINT_NAMES[i]}: plan range "
                             f"[{view.q_plan_min[i]:.1f}, {view.q_plan_max[i]:.1f}]  "
                             f"(true [{cal.q_min[i]:.1f}, {cal.q_max[i]:.1f}], "
                             f"remap {view.remap_offset[i]:+.1f})")
                if not (view.q_plan_min[i] - 1e-6 <= q0[i] <= view.q_plan_max[i] + 1e-6):
                    lines.append(f"  ! q0 for {JOINT_NAMES[i]} out of range"); bad = True
                if not (view.q_plan_min[i] - 1e-6 <= qf[i] <= view.q_plan_max[i] + 1e-6):
                    lines.append(f"  ! qf for {JOINT_NAMES[i]} out of range"); bad = True

            if bad:
                self.status.insert(tk.END, "\n".join(lines))
                return

            traj = LSPBTrajectory(n_joints=5)
            tf = traj.plan(q0, qf, vmax, amax)

            dt = tf / 300
            t = np.arange(0, tf + dt, dt)
            ideal = np.zeros((5, len(t)))
            quant = np.zeros((5, len(t)))
            res = self.resolution.get()

            for k, ti in enumerate(t):
                q_plan = traj.get_state(ti)
                q_true = view.to_true(q_plan)
                s = cal.dh_to_servo(q_true)
                ideal[:, k] = s
                quant[:, k] = np.round(s / res) * res

            max_err = np.max(np.abs(ideal - quant), axis=1)
            lines.append(f"\nMove time: {tf:.3f} s")
            lines.append("Max quantization error (deg): " +
                          ", ".join(f"{e:.3f}" for e in max_err))
            self.status.insert(tk.END, "\n".join(lines))

            for j in range(5):
                ax = self.axs[j]
                ax.clear()
                ax.plot(t, ideal[j], color="tab:blue", linewidth=1.2, label="Ideal")
                ax.step(t, quant[j], color="tab:red", linewidth=1.0,
                        where="post", label="Quantized")
                ax.set_ylabel(JOINT_NAMES[j])
                ax.grid(True, alpha=0.3)
            self.axs[0].legend(loc="lower right", fontsize=7)
            self.axs[-1].set_xlabel("Time (s)")
            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            self.status.insert(tk.END, f"Error: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = LSPBApp(root)
    root.mainloop()