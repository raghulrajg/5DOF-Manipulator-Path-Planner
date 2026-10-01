import tkinter as tk
from tkinter import ttk
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ======================================================================
# Core math
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
        qd = np.zeros(self.n)
        qdd = np.zeros(self.n)
        ti = min(max(t, 0.0), self.tf)
        
        for j in range(self.n):
            q0, qf, tb, a, v, tf = (self.q0[j], self.qf[j], self.tb[j],
                                     self.a[j], self.v[j], self.tf)
            if qf == q0:
                q[j] = q0; qd[j] = 0.0; qdd[j] = 0.0
                continue
            
            if ti <= tb:
                q[j] = q0 + 0.5*a*ti**2
                qd[j] = a * ti
                qdd[j] = a
            elif ti <= tf - tb:
                q[j] = q0 + a*tb*(ti - tb/2)
                qd[j] = a * tb
                qdd[j] = 0.0
            else:
                q[j] = qf - 0.5*a*(tf-ti)**2
                qd[j] = a * (tf - ti)
                qdd[j] = -a
                
        return q, qd, qdd


class MultiSegmentLSPB:
    """Chains multiple LSPB trajectories together to pass through Via Points"""
    def __init__(self, n_joints):
        self.n = n_joints
        self.segments = []
        self.via_times = []
        self.total_tf = 0.0

    def plan(self, waypoints_list, vmax, amax, desired_total_time=0.0, min_tf=0.05):
        self.segments = []
        self.via_times = [0.0]
        self.total_tf = 0.0
        
        # Pass 1: Compute minimum time needed for each segment
        min_seg_times = []
        for i in range(len(waypoints_list) - 1):
            q_start = waypoints_list[i]
            q_end = waypoints_list[i+1]
            seg = LSPBTrajectory(self.n)
            tf = seg.plan(q_start, q_end, vmax, amax, min_tf)
            min_seg_times.append(tf)

        total_min_time = sum(min_seg_times)
        
        # Determine Scaling Factor if user provided a specific total time
        scale = 1.0
        time_warning = False
        if desired_total_time > total_min_time:
            scale = desired_total_time / total_min_time
        elif desired_total_time > 0 and desired_total_time < total_min_time:
            # Cannot shrink below physical limits (vmax/amax)
            time_warning = True

        # Pass 2: Plan segments with the scaled enforced times
        for i in range(len(waypoints_list) - 1):
            q_start = waypoints_list[i]
            q_end = waypoints_list[i+1]
            seg = LSPBTrajectory(self.n)
            
            enforced_tf = min_seg_times[i] * scale
            tf = seg.plan(q_start, q_end, vmax, amax, min_tf=enforced_tf)
            
            self.segments.append(seg)
            self.total_tf += tf
            self.via_times.append(self.total_tf)
            
        return self.total_tf, time_warning

    def get_state(self, t):
        if t <= 0:
            return self.segments[0].get_state(0.0)
        if t >= self.total_tf:
            return self.segments[-1].get_state(self.segments[-1].tf)
            
        for i, seg in enumerate(self.segments):
            start_t = self.via_times[i]
            end_t = self.via_times[i+1]
            if start_t <= t <= end_t:
                return seg.get_state(t - start_t)
        
        return self.segments[-1].get_state(self.segments[-1].tf)


# ======================================================================
# GUI
# ======================================================================
JOINT_NAMES = ["q1", "q2", "q3", "q4", "q5"]
MAX_VIAS = 20  

DEFAULTS = dict(
    offset=[0, 90, 90, 90, 0],
    direction=[1, 1, 1, 1, 1],
    no_remap=[False, False, False, False, True],
    vmax=[60, 60, 60, 90, 90],
    amax=[120, 120, 120, 180, 180],
    via_paths=[
        [0, 45, 90, 0],
        [0, -30, -10, 0],
        [0, 60, 30, 0],
        [0, 10, 20, 0],
        [0, 60, 45, 0]
    ]
)

class LSPBApp:
    def __init__(self, root):
        self.root = root
        root.title("LSPB Multi-Segment Trajectory Planner")
        root.geometry("1200x850") 

        # ---------------------------------------------------------
        # Setting up a Scrollable Canvas (Vertical & Horizontal)
        # ---------------------------------------------------------
        self.main_canvas = tk.Canvas(root, highlightthickness=0)
        self.v_scroll = ttk.Scrollbar(root, orient="vertical", command=self.main_canvas.yview)
        self.h_scroll = ttk.Scrollbar(root, orient="horizontal", command=self.main_canvas.xview)
        
        self.scrollable_frame = ttk.Frame(self.main_canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(
                scrollregion=self.main_canvas.bbox("all")
            )
        )

        self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.main_canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)

        self.h_scroll.pack(side="bottom", fill="x")
        self.v_scroll.pack(side="right", fill="y")
        self.main_canvas.pack(side="left", fill="both", expand=True)
        
        root.bind_all("<MouseWheel>", self._on_mousewheel) 
        root.bind_all("<Button-4>", self._on_mousewheel)   
        root.bind_all("<Button-5>", self._on_mousewheel)   
        
        # ---------------------------------------------------------
        # Initialize Variables
        # ---------------------------------------------------------
        self.vars = {
            "offset": [tk.DoubleVar(value=DEFAULTS["offset"][i]) for i in range(5)],
            "direction": [tk.StringVar(value=str(DEFAULTS["direction"][i])) for i in range(5)],
            "no_remap": [tk.BooleanVar(value=DEFAULTS["no_remap"][i]) for i in range(5)],
            "vmax": [tk.DoubleVar(value=DEFAULTS["vmax"][i]) for i in range(5)],
            "amax": [tk.DoubleVar(value=DEFAULTS["amax"][i]) for i in range(5)],
        }
        
        self.via_vars = []
        for i in range(5):
            row = []
            for j in range(MAX_VIAS):
                val = DEFAULTS["via_paths"][i][j] if j < len(DEFAULTS["via_paths"][i]) else 0.0
                row.append(tk.DoubleVar(value=val))
            self.via_vars.append(row)
            
        self.num_vias_var = tk.IntVar(value=len(DEFAULTS["via_paths"][0]))

        # ---------------------------------------------------------
        # UI Layout Construction
        # ---------------------------------------------------------
        # Top Config Frame
        top_frame = ttk.Frame(self.scrollable_frame, padding=10)
        top_frame.grid(row=0, column=0, sticky="w")
        ttk.Label(top_frame, text="Number of Via Points:").pack(side="left")
        ttk.Spinbox(top_frame, from_=2, to=MAX_VIAS, textvariable=self.num_vias_var, width=5).pack(side="left", padx=5)
        ttk.Button(top_frame, text="Update Table Columns", command=self.build_table).pack(side="left", padx=10)
        
        # Dynamic Table Frame
        self.table_frame = ttk.Frame(self.scrollable_frame, padding=(10, 0))
        self.table_frame.grid(row=1, column=0, sticky="w")
        
        # Global Parameters Frame
        gframe = ttk.Frame(self.scrollable_frame, padding=(10, 10))
        gframe.grid(row=2, column=0, sticky="w")
        
        self.servo_min = tk.DoubleVar(value=0)
        self.servo_max = tk.DoubleVar(value=180)
        self.resolution = tk.DoubleVar(value=1.0)
        self.num_samples = tk.IntVar(value=40) 
        self.desired_time = tk.DoubleVar(value=0.0) # Added Desired Total Time
        
        inputs = [("Servo min", self.servo_min),
                  ("Servo max", self.servo_max),
                  ("Resolution (deg)", self.resolution),
                  ("Time Steps (Dots)", self.num_samples),
                  ("Desired Total Time (s) [0=Auto]", self.desired_time)]
                  
        for label, var in inputs:
            ttk.Label(gframe, text=label).pack(side="left", padx=(0, 4))
            ttk.Entry(gframe, textvariable=var, width=6).pack(side="left", padx=(0, 12))

        ttk.Button(gframe, text="Plan & Plot", command=self.plan_and_plot).pack(side="left", padx=15)

        # Status text
        self.status = tk.Text(self.scrollable_frame, height=6, width=120, font=("Courier", 9))
        self.status.grid(row=3, column=0, padx=10, pady=(6, 6), sticky="w")

        # Matplotlib figure
        self.fig = Figure(figsize=(12, 10)) 
        self.axs = self.fig.subplots(5, 3, sharex=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.scrollable_frame)
        self.canvas.get_tk_widget().grid(row=4, column=0, padx=10, pady=(0, 10))

        self.build_table()
        self.plan_and_plot()

    def build_table(self):
        for w in self.table_frame.winfo_children():
            w.destroy()
            
        n_vias = self.num_vias_var.get()
        if n_vias < 2:
            n_vias = 2
            self.num_vias_var.set(2)
        elif n_vias > MAX_VIAS:
            n_vias = MAX_VIAS
            self.num_vias_var.set(MAX_VIAS)
            
        headers = ["Joint", "Offset", "Dir", "No-remap"] + [f"WP {j+1}" for j in range(n_vias)] + ["vmax", "amax"]
        
        for c, h in enumerate(headers):
            ttk.Label(self.table_frame, text=h, font=("", 9, "bold")).grid(row=0, column=c, padx=4)

        for i in range(5):
            ttk.Label(self.table_frame, text=JOINT_NAMES[i]).grid(row=i+1, column=0)
            ttk.Entry(self.table_frame, textvariable=self.vars["offset"][i], width=6).grid(row=i+1, column=1)
            ttk.Combobox(self.table_frame, textvariable=self.vars["direction"][i], values=["1", "-1"],
                         width=4, state="readonly").grid(row=i+1, column=2)
            ttk.Checkbutton(self.table_frame, variable=self.vars["no_remap"][i]).grid(row=i+1, column=3)
            
            col_offset = 4
            for j in range(n_vias):
                ttk.Entry(self.table_frame, textvariable=self.via_vars[i][j], width=5).grid(row=i+1, column=col_offset+j, padx=2)
                
            ttk.Entry(self.table_frame, textvariable=self.vars["vmax"][i], width=6).grid(row=i+1, column=col_offset+n_vias, padx=2)
            ttk.Entry(self.table_frame, textvariable=self.vars["amax"][i], width=6).grid(row=i+1, column=col_offset+n_vias+1, padx=2)

    def _on_mousewheel(self, event):
        if event.num == 4 or getattr(event, "delta", 0) > 0:
            self.main_canvas.yview_scroll(-1, "units")
        elif event.num == 5 or getattr(event, "delta", 0) < 0:
            self.main_canvas.yview_scroll(1, "units")

    def _read(self):
        offset = [v.get() for v in self.vars["offset"]]
        direction = [float(v.get()) for v in self.vars["direction"]]
        no_remap = [i for i, v in enumerate(self.vars["no_remap"]) if v.get()]
        vmax = [v.get() for v in self.vars["vmax"]]
        amax = [v.get() for v in self.vars["amax"]]
        
        n_vias = self.num_vias_var.get()
        paths = []
        for i in range(5):
            pts = [self.via_vars[i][j].get() for j in range(n_vias)]
            paths.append(pts)
            
        waypoints_list = np.array(paths).T.tolist()
        
        return offset, direction, no_remap, waypoints_list, vmax, amax

    def plan_and_plot(self):
        self.status.delete("1.0", tk.END)
        try:
            offset, direction, no_remap, waypoints_list, vmax, amax = self._read()

            cal = ServoCalibration(offset, direction,
                                    self.servo_min.get(), self.servo_max.get())
            view = SymmetricPlanningView(cal, no_remap_joints=no_remap)

            lines = []
            bad = False
            
            for pt_idx, wp in enumerate(waypoints_list):
                for i in range(5):
                    if not (view.q_plan_min[i] - 1e-6 <= wp[i] <= view.q_plan_max[i] + 1e-6):
                        lines.append(f"  ! Joint {JOINT_NAMES[i]} via point WP {pt_idx+1} ({wp[i]}) out of bounds.")
                        bad = True

            if bad:
                self.status.insert(tk.END, "\n".join(lines))
                return

            traj = MultiSegmentLSPB(n_joints=5)
            
            # Pass desired total time from GUI to the planner
            desired_t = self.desired_time.get()
            tf, time_warning = traj.plan(waypoints_list, vmax, amax, desired_total_time=desired_t)

            if time_warning:
                lines.append(f"WARNING: Desired time ({desired_t}s) is too fast for the physical limits (vmax/amax).")
                lines.append(f"The planner clamped the trajectory to the fastest possible safe time ({tf:.3f} s).")

            n_samples = max(2, self.num_samples.get())
            t_hr = np.linspace(0, tf, max(500, int(tf * 100)))
            t_wp = np.linspace(0, tf, n_samples)
            res = self.resolution.get()

            def compute_states(time_array):
                q_arr = np.zeros((5, len(time_array)))
                qd_arr = np.zeros((5, len(time_array)))
                qdd_arr = np.zeros((5, len(time_array)))
                
                for k, ti in enumerate(time_array):
                    q_plan, qd_plan, qdd_plan = traj.get_state(ti)
                    q_true = view.to_true(q_plan)
                    
                    q_arr[:, k] = cal.dh_to_servo(q_true)
                    qd_arr[:, k] = cal.directions * qd_plan
                    qdd_arr[:, k] = cal.directions * qdd_plan
                return q_arr, qd_arr, qdd_arr

            ideal_q_hr, ideal_qd_hr, ideal_qdd_hr = compute_states(t_hr)
            ideal_q_wp, ideal_qd_wp, ideal_qdd_wp = compute_states(t_wp)
            quant_q_wp = np.round(ideal_q_wp / res) * res

            max_err = np.max(np.abs(ideal_q_wp - quant_q_wp), axis=1)
            lines.append(f"Total move time: {tf:.3f} s | Number of Segments: {len(waypoints_list)-1}")
            lines.append("Max quantization error (deg): " + ", ".join(f"{e:.3f}" for e in max_err))
            self.status.insert(tk.END, "\n".join(lines))

            # Set Master Title for Total Time
            self.fig.suptitle(f"Total Trajectory Time: {tf:.3f} Seconds", fontsize=14, fontweight='bold', color='navy')

            for j in range(5):
                ax_pos = self.axs[j, 0]
                ax_vel = self.axs[j, 1]
                ax_acc = self.axs[j, 2]
                
                ax_pos.clear(); ax_vel.clear(); ax_acc.clear()

                ax_pos.plot(t_hr, ideal_q_hr[j], color="tab:blue", linewidth=1.2, label="Continuous Path")
                ax_pos.plot(t_wp, ideal_q_wp[j], marker='o', markersize=3, linestyle='None', color="black", alpha=0.5, label="Time Steps")
                ax_pos.step(t_wp, quant_q_wp[j], color="tab:red", linewidth=1.0, where="post", label="Quantized")
                ax_pos.set_ylabel(f"{JOINT_NAMES[j]} Pos")
                ax_pos.grid(True, alpha=0.3)

                ax_vel.plot(t_hr, ideal_qd_hr[j], color="tab:orange", linewidth=1.2)
                ax_vel.set_ylabel(f"Vel (deg/s)")
                ax_vel.grid(True, alpha=0.3)

                ax_acc.plot(t_hr, ideal_qdd_hr[j], color="tab:green", linewidth=1.2)
                ax_acc.set_ylabel(f"Acc (deg/s²)")
                ax_acc.grid(True, alpha=0.3)
                
                # Draw vertical lines for via points
                for via_t in traj.via_times:
                    ax_pos.axvline(via_t, color='gray', linestyle='--', alpha=0.5)
                    ax_vel.axvline(via_t, color='gray', linestyle='--', alpha=0.5)
                    ax_acc.axvline(via_t, color='gray', linestyle='--', alpha=0.5)
                
                if j == 0:
                    ax_pos.set_title("Position (Degrees)")
                    ax_vel.set_title("Velocity (Deg/s)")
                    ax_acc.set_title("Acceleration (Deg/s²)")

            self.axs[0, 0].legend(loc="best", fontsize=7)
            
            # Format X-axis to explicitly show timestamps of via points
            for col in range(3):
                self.axs[-1, col].set_xlabel("Time (s)", fontweight='bold')
                self.axs[-1, col].set_xticks(traj.via_times)
                self.axs[-1, col].set_xticklabels([f"{t:.2f}" for t in traj.via_times], rotation=45, fontsize=8)

            # Adjust layout to make room for the big master title at the top
            self.fig.tight_layout(rect=[0, 0.02, 1, 0.96])
            self.canvas.draw()

        except Exception as e:
            self.status.insert(tk.END, f"Error: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = LSPBApp(root)
    root.mainloop()