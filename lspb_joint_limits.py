import numpy as np

# ======================================================================
# 1. Servo calibration layer
# ======================================================================
class ServoCalibration:
    def __init__(self, offsets, directions=None, servo_min=0.0, servo_max=180.0):
        """
        offsets    : list, servo angle (deg) that corresponds to q_DH = 0,
                     per joint. e.g. [0, 90, 90, 90, 0]
        directions : list of +1/-1 per joint (default: all +1)
        servo_min/servo_max : physical servo command range (deg)
        """
        self.offsets = np.asarray(offsets, dtype=float)
        self.n = len(offsets)
        self.directions = (np.ones(self.n) if directions is None
                            else np.asarray(directions, dtype=float))
        self.servo_min = servo_min
        self.servo_max = servo_max

        # Derive DH-space joint limits from the servo's physical range
        self.q_min = np.zeros(self.n)
        self.q_max = np.zeros(self.n)
        for i in range(self.n):
            d = self.directions[i]
            o = self.offsets[i]
            # servo = d*q + o  =>  q = (servo - o) / d
            lim_a = (self.servo_min - o) / d
            lim_b = (self.servo_max - o) / d
            self.q_min[i] = min(lim_a, lim_b)
            self.q_max[i] = max(lim_a, lim_b)

    def dh_to_servo(self, q_dh, clamp=True):
        """Convert DH joint angles -> servo command angles."""
        q_dh = np.asarray(q_dh, dtype=float)
        servo = self.directions * q_dh + self.offsets
        if clamp:
            servo = np.clip(servo, self.servo_min, self.servo_max)
        return servo

    def servo_to_dh(self, servo):
        """Convert servo command angles -> DH joint angles."""
        servo = np.asarray(servo, dtype=float)
        return (servo - self.offsets) / self.directions

    def check_within_limits(self, q_dh, joint_names=None):
        """Raise ValueError listing any joint(s) outside DH-space limits."""
        q_dh = np.asarray(q_dh, dtype=float)
        bad = []
        for i in range(self.n):
            if not (self.q_min[i] - 1e-6 <= q_dh[i] <= self.q_max[i] + 1e-6):
                name = joint_names[i] if joint_names else f"q{i+1}"
                bad.append(f"{name}={q_dh[i]:.2f} (allowed "
                            f"[{self.q_min[i]:.2f}, {self.q_max[i]:.2f}])")
        if bad:
            raise ValueError("Joint limit violation: " + "; ".join(bad))

    def print_limits(self, joint_names=None):
        for i in range(self.n):
            name = joint_names[i] if joint_names else f"q{i+1}"
            print(f"{name}: DH range [{self.q_min[i]:7.2f}, {self.q_max[i]:7.2f}] deg "
                  f"(offset={self.offsets[i]}, dir={self.directions[i]:+.0f})")


# ======================================================================
# 2. Joint-space LSPB (same as before) + limit-aware planning
# ======================================================================
class LSPBTrajectory:
    def __init__(self, n_joints, calibration=None, joint_names=None):
        self.n = n_joints
        self.cal = calibration          # ServoCalibration instance or None
        self.joint_names = joint_names
        self.q0 = self.qf = self.tb = self.a = self.v = self.tf = None
        self._planned = False

    def _min_time_single_joint(self, h, vmax, amax):
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
            t_cruise = (h - dist_accel_decel) / vmax
            tf = 2 * tb + t_cruise
        return tf, tb

    def plan(self, q0, qf, vmax, amax, tf_override=None, min_tf=0.05):
        q0 = np.asarray(q0, dtype=float)
        qf = np.asarray(qf, dtype=float)
        vmax = np.asarray(vmax, dtype=float)
        amax = np.asarray(amax, dtype=float)

        # --- Joint-limit check BEFORE planning (fail fast, not mid-motion) ---
        if self.cal is not None:
            self.cal.check_within_limits(q0, self.joint_names)
            self.cal.check_within_limits(qf, self.joint_names)
            # Note: LSPB moves monotonically from q0 to qf with no overshoot,
            # so checking both endpoints guarantees the whole path stays
            # within limits -- no need to check intermediate points.

        joint_min_tf = np.array([
            self._min_time_single_joint(qf[j] - q0[j], vmax[j], amax[j])[0]
            for j in range(self.n)
        ])
        tf = max(joint_min_tf.max(), min_tf)
        if tf_override is not None:
            if tf_override < joint_min_tf.max():
                raise ValueError(f"tf_override infeasible; need >= {joint_min_tf.max():.3f}s")
            tf = tf_override

        tb = np.zeros(self.n); a = np.zeros(self.n); v = np.zeros(self.n)
        for j in range(self.n):
            h = qf[j] - q0[j]
            if h == 0:
                tb[j] = tf / 4
                continue
            aj = amax[j] * np.sign(h)
            disc = aj**2 * tf**2 - 4 * aj * h
            tb[j] = tf / 2 if disc < 0 else np.clip(tf/2 - np.sqrt(disc)/(2*aj), 1e-6, tf/2)
            a[j] = h / (tb[j] * (tf - tb[j]))
            v[j] = a[j] * tb[j]

        self.q0, self.qf, self.tb, self.a, self.v, self.tf = q0, qf, tb, a, v, tf
        self._planned = True
        return tf

    def get_state(self, t):
        if not self._planned:
            raise RuntimeError("Call plan() first.")
        q = np.zeros(self.n); qd = np.zeros(self.n); qdd = np.zeros(self.n)
        ti = min(max(t, 0.0), self.tf)
        done = t >= self.tf
        for j in range(self.n):
            q0, qf, tb, a, v, tf = self.q0[j], self.qf[j], self.tb[j], self.a[j], self.v[j], self.tf
            if qf == q0:
                q[j] = q0; continue
            if ti <= tb:
                q[j], qd[j], qdd[j] = q0 + 0.5*a*ti**2, a*ti, a
            elif ti <= tf - tb:
                q[j], qd[j], qdd[j] = q0 + a*tb*(ti - tb/2), v, 0.0
            else:
                q[j], qd[j], qdd[j] = qf - 0.5*a*(tf-ti)**2, a*(tf-ti), -a

        # Safety net: clamp to DH limits before returning (should already
        # be satisfied by construction, but cheap insurance against
        # floating point edge cases at the very last sample).
        if self.cal is not None:
            q = np.clip(q, self.cal.q_min, self.cal.q_max)

        return q, qd, qdd, done