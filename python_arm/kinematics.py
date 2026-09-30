import numpy as np
from scipy.optimize import least_squares


def dh_transformation_matrix(alpha, a, d, theta):
    """
    Computes the standard Denavit-Hartenberg (DH) homogeneous transformation matrix.

    Parameters:
    alpha : float - Link twist (radians)
    a     : float - Link length
    d     : float - Link offset
    theta : float - Joint angle (radians)
    """
    return np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta),  np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0,              np.sin(alpha),                  np.cos(alpha),                 d],
        [0,              0,                              0,                             1]
    ])


def get_effective_transformation(dh_table):
    """
    Computes the overall effective transformation matrix by multiplying joint matrices.
    """
    T_eff = np.eye(4)
    for row in dh_table:
        alpha, a, d, theta = row
        T_next = dh_transformation_matrix(alpha, a, d, theta)
        T_eff = np.dot(T_eff, T_next)
    return T_eff


def build_dh_table(q):
    """
    Builds the 5-row DH table for the manipulator from joint angles q
    (radians). This is the UPDATED DH table -- no qh helper joint;
    q = [0,0,0,0,0] corresponds to hardware home
    (servo = [90,180,180,90,0]).

    Joint limits (degrees), for reference:
        theta1: -90 to 90
        theta2: -180 to 0
        theta3: 0 to 180
        theta4: -90 to 90
        theta5: 0 to 180
    """
    theta1, theta2, theta3, theta4, theta5 = q

    return [
        #    alpha,           a,     d,     theta
        [np.radians(90),      0,    0.04,  theta1],                      # Joint 1
        [       0,          0.199,    0,   theta2 + np.radians(168.4)],  # Joint 2
        [       0,          0.14,     0,   theta3 - np.radians(168.4)],  # Joint 3
        [np.radians(90),      0,      0,   theta4 + np.radians(90)],     # Joint 4
        [       0,            0,    0.126, theta5]                       # Joint 5
    ]


def get_forward_kinematics(q):
    """
    Computes the Forward Kinematics for the 5-DOF robotic arm using the
    updated DH table (no qh helper joint).
    """
    return get_effective_transformation(build_dh_table(q))


def numeric_jacobian(q):
    """
    Calculates the 6x5 numeric Jacobian matrix.
    """
    delta = 1e-5
    J = np.zeros((6, len(q)))
    T_base = get_forward_kinematics(q)

    for i in range(len(q)):
        q_step = np.copy(q)
        q_step[i] += delta
        T_step = get_forward_kinematics(q_step)

        dp = (T_step[0:3, 3] - T_base[0:3, 3]) / delta

        R_diff = T_step[0:3, 0:3] @ T_base[0:3, 0:3].T
        dw = np.array([
            R_diff[2, 1] - R_diff[1, 2],
            R_diff[0, 2] - R_diff[2, 0],
            R_diff[1, 0] - R_diff[0, 1]
        ]) / (2 * delta)

        J[:, i] = np.concatenate((dp, dw))

    return J


def inverse_kinematics_dls(T_target, q_guess, max_iter=1000,
                            lambda_factor=0.1, tol=1e-6):
    """
    Jacobian Damped Least Squares Inverse Kinematics solver (UNCONSTRAINED
    -- does not respect joint limits). Kept for reference/comparison;
    prefer inverse_kinematics_optimized() below for actual use.
    """
    q = np.array(q_guess, dtype=float)

    for _ in range(max_iter):
        T_curr = get_forward_kinematics(q)
        p_err = T_target[0:3, 3] - T_curr[0:3, 3]
        R_diff = T_target[0:3, 0:3] @ T_curr[0:3, 0:3].T
        w_err = np.array([
            R_diff[2, 1] - R_diff[1, 2],
            R_diff[0, 2] - R_diff[2, 0],
            R_diff[1, 0] - R_diff[0, 1]
        ]) / 2.0

        err_vec = np.concatenate((p_err, w_err))
        if np.linalg.norm(err_vec) < tol:
            break

        J = numeric_jacobian(q)
        H = J.T @ J + (lambda_factor**2) * np.eye(len(q))
        gradient = J.T @ err_vec
        dq = np.linalg.solve(H, gradient)
        q += dq

    q = (q + np.pi) % (2 * np.pi) - np.pi
    return q


def _pose_residuals(q, T_target, w_pos=1.0, w_orient=1.0):
    """
    Residual vector (not a scalar) for least_squares: 3 position residuals
    + 9 rotation-matrix-difference residuals. Using the Frobenius-norm-
    style residual (raw matrix difference) avoids the axis-angle vector-
    part blind spot at 180-degree orientation errors (sin(180)=0).
    """
    T = get_forward_kinematics(q)
    p_err = (T[0:3, 3] - T_target[0:3, 3]) * np.sqrt(w_pos)
    R_err = (T[0:3, 0:3] - T_target[0:3, 0:3]).flatten() * np.sqrt(w_orient)
    return np.concatenate([p_err, R_err])


def inverse_kinematics_optimized(T_target, bounds_rad,
                                  n_starts=8, seed=0,
                                  w_pos=1.0, w_orient=1.0,
                                  primary_guess=None):
    """
    Bounded nonlinear-least-squares Inverse Kinematics: minimizes the sum
    of squared position + orientation residuals subject to joint-limit
    bounds using scipy.optimize.least_squares (trust-region reflective),
    which respects the bounds by construction and converges reliably
    even near joint-limit boundaries.

    Parameters:
        T_target  : 4x4 target pose
        bounds_rad: list of (min, max) tuples in RADIANS, one per joint
        n_starts  : number of random starting points to try, in addition
                    to the midpoint of the bounds
        w_pos, w_orient: relative weight of position vs orientation
                    residuals (equal by default)
        primary_guess: optional joint angles (radians) to try FIRST --
                    biases the solver toward a specific configuration
                    when the arm is redundant for a given pose (multiple
                    joint sets can reach the same pose).

    Returns: (q_solved_rad, cost, success)
        cost is the final sum-of-squared-residuals (near 0 = good fit).
        success is True only if the residual norm is below a sane
        threshold -- always check this before trusting q.
    """
    bounds_rad = list(bounds_rad)
    lo = np.array([b[0] for b in bounds_rad])
    hi = np.array([b[1] for b in bounds_rad])

    rng = np.random.default_rng(seed)
    starts = []
    if primary_guess is not None:
        starts.append(np.array(primary_guess, dtype=float))
    starts.append((lo + hi) / 2.0)
    for _ in range(n_starts):
        starts.append(rng.uniform(lo, hi))

    best = None  # (cost, q_sol)
    for q0 in starts:
        q0_clipped = np.clip(q0, lo, hi)
        res = least_squares(_pose_residuals, q0_clipped,
                             args=(T_target, w_pos, w_orient),
                             bounds=(lo, hi), method="trf")
        cost = np.sum(res.fun**2)
        if best is None or cost < best[0]:
            best = (cost, res.x)
        if primary_guess is not None and cost < 1e-10:
            break

    cost, q_sol = best
    success = bool(cost < 1e-6)
    return q_sol, cost, success