"""
kinematics.py

Forward and inverse kinematics for the 5-DOF robotic arm.

DH parameters are calibrated so that q = [0, 0, 0, 0, 0] corresponds
directly to hardware home (servo [90, 180, 180, 90, 0]) -- there is no
longer a separate "kin space" vs "hardware space" offset to track.

Joint limits (deg), from the arm's mechanical constraints:
    q1: -90 to  90
    q2:   0 to 180
    q3:   0 to 180
    q4: -90 to  90
    q5: -180 to 180
"""

import numpy as np
from scipy.optimize import least_squares

# Joint limits, degrees -- (min, max) per joint, same q convention as
# get_forward_kinematics. Used as hard bounds during IK optimization.
JOINT_LIMITS_DEG = [
    (-90.0,  90.0),   # q1
    (0.0,   180.0),   # q2
    (0.0,   180.0),   # q3
    (-90.0,  90.0),   # q4
    (-180.0, 180.0),  # q5
]


def get_forward_kinematics(q, qh=0.0):
    """
    Computes the Forward Kinematics for the 5-DOF robotic arm.
    q = [0,0,0,0,0] corresponds to hardware home (servo [90,180,180,90,0]).
    """
    q1, q2, q3, q4, q5 = q

    # DH Parameters [a (meters), alpha (radians), d (meters), theta (radians)]
    dh_params = [
        [0.0,    np.pi/2, 0.04,  q1],
        [0.195,  0.0,     0.0,   q2 + np.pi],
        [0.04,   0.0,     0.0,   qh - np.pi/2],
        [0.14,   0.0,     0.0,   q3 - np.pi/2],
        [0.0,    np.pi/2, 0.0,   q4 + np.pi/2],
        [0.0,    0.0,     0.126, q5]
    ]

    T_effective = np.eye(4)

    for a, alpha, d, theta in dh_params:
        A = np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0.0,            np.sin(alpha),                np.cos(alpha),               d],
            [0.0,            0.0,                          0.0,                         1.0]
        ])
        T_effective = T_effective @ A

    return T_effective


def numeric_jacobian(q, qh=0.0):
    """
    Calculates the 6x5 numeric Jacobian matrix.
    """
    delta = 1e-5
    J = np.zeros((6, len(q)))
    T_base = get_forward_kinematics(q, qh)

    for i in range(len(q)):
        q_step = np.copy(q)
        q_step[i] += delta
        T_step = get_forward_kinematics(q_step, qh)

        dp = (T_step[0:3, 3] - T_base[0:3, 3]) / delta

        R_diff = T_step[0:3, 0:3] @ T_base[0:3, 0:3].T
        dw = np.array([
            R_diff[2, 1] - R_diff[1, 2],
            R_diff[0, 2] - R_diff[2, 0],
            R_diff[1, 0] - R_diff[0, 1]
        ]) / (2 * delta)

        J[:, i] = np.concatenate((dp, dw))

    return J


def _pose_error(q, T_target, qh=0.0):
    """
    6-vector pose error: [position error (3); axis-angle orientation
    error (3)]. Used as the residual for the least-squares IK optimizer.
    """
    T_curr = get_forward_kinematics(q, qh)

    p_err = T_target[0:3, 3] - T_curr[0:3, 3]

    R_diff = T_target[0:3, 0:3] @ T_curr[0:3, 0:3].T
    w_err = np.array([
        R_diff[2, 1] - R_diff[1, 2],
        R_diff[0, 2] - R_diff[2, 0],
        R_diff[1, 0] - R_diff[0, 1]
    ]) / 2.0

    return np.concatenate((p_err, w_err))


def inverse_kinematics_opt(T_target, q_guess, qh=0.0, joint_limits_deg=None):
    """
    Optimization-based Inverse Kinematics.

    Minimizes the 6-DOF pose error (position + axis-angle orientation)
    using scipy.optimize.least_squares -- a bounded trust-region-
    reflective nonlinear least-squares solver. Joint limits are enforced
    directly as bounds *during* the search, so the returned solution is
    guaranteed to be within limits (rather than being solved unconstrained
    and checked/clamped afterward).

    Parameters
    ----------
    T_target : (4,4) ndarray -- desired end-effector pose
    q_guess  : (5,) array-like, radians -- initial guess for the optimizer
    qh       : float, radians -- fixed auxiliary joint angle
    joint_limits_deg : optional list of (min_deg, max_deg) per joint;
                        defaults to JOINT_LIMITS_DEG

    Returns
    -------
    q_solution_rad : (5,) ndarray, radians
    result : scipy.optimize.OptimizeResult (inspect .success, .cost,
             .status for diagnostics)
    """
    limits = joint_limits_deg if joint_limits_deg is not None else JOINT_LIMITS_DEG
    lower = np.deg2rad(np.array([lo for lo, hi in limits]))
    upper = np.deg2rad(np.array([hi for lo, hi in limits]))

    q0 = np.clip(np.array(q_guess, dtype=float), lower, upper)

    result = least_squares(
        _pose_error,
        x0=q0,
        bounds=(lower, upper),
        args=(T_target, qh),
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
        max_nfev=5000
    )
    return result.x, result


def inverse_kinematics_dls(T_target, q_guess, qh=0.0, max_iter=1000, lambda_factor=0.1, tol=1e-6):
    """
    Jacobian Damped Least Squares Inverse Kinematics solver (legacy,
    unconstrained -- kept for compatibility / comparison. Does NOT
    enforce joint limits; prefer inverse_kinematics_opt for planning
    real moves since it respects JOINT_LIMITS_DEG).
    """
    q = np.array(q_guess, dtype=float)

    for _ in range(max_iter):
        T_curr = get_forward_kinematics(q, qh)

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

        J = numeric_jacobian(q, qh)

        H = J.T @ J + (lambda_factor**2) * np.eye(len(q))
        gradient = J.T @ err_vec
        dq = np.linalg.solve(H, gradient)

        q += dq

    q = (q + np.pi) % (2 * np.pi) - np.pi

    return q