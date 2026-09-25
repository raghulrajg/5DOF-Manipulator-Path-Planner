import numpy as np
from scipy.optimize import least_squares


def get_forward_kinematics(q, qh=0.0):
    """
    Computes the Forward Kinematics for the 5-DOF robotic arm, using the
    UPDATED DH table (built so that q = [0,0,0,0,0] corresponds directly
    to hardware home, servo = [90,180,180,90,0]).
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


def inverse_kinematics_dls(T_target, q_guess, qh=0.0, max_iter=1000,
                            lambda_factor=0.1, tol=1e-6):
    """
    Jacobian Damped Least Squares Inverse Kinematics solver (UNCONSTRAINED
    -- does not respect joint limits). Kept for reference/comparison;
    prefer inverse_kinematics_optimized() below for actual use, since it
    enforces joint limits directly instead of needing post-hoc rejection.
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


def _pose_residuals(q, T_target, qh, w_pos=1.0, w_orient=1.0):
    """
    Residual vector (not a scalar) for least_squares: 3 position residuals
    + 9 rotation-matrix-difference residuals. least_squares minimizes the
    sum of squares of ALL these components using a proper nonlinear
    least-squares algorithm (trust-region reflective), which is far more
    reliable at actually reaching zero residual (when a solution exists)
    than treating pose-matching as a single scalar cost for a general
    bounded minimizer (L-BFGS-B) -- especially near joint-limit
    boundaries, where a scalar optimizer's gradient can get clipped
    before it finds the exact solution.
    """
    T = get_forward_kinematics(q, qh)
    p_err = (T[0:3, 3] - T_target[0:3, 3]) * np.sqrt(w_pos)
    R_err = (T[0:3, 0:3] - T_target[0:3, 0:3]).flatten() * np.sqrt(w_orient)
    return np.concatenate([p_err, R_err])


def inverse_kinematics_optimized(T_target, bounds_rad, qh=0.0,
                                  n_starts=8, seed=0,
                                  w_pos=1.0, w_orient=1.0,
                                  primary_guess=None):
    """
    Bounded nonlinear-least-squares Inverse Kinematics: minimizes the sum
    of squared position + orientation residuals subject to joint-limit
    bounds using scipy.optimize.least_squares (trust-region reflective),
    which respects the bounds by construction. Prefer this over a scalar
    bounded minimizer (L-BFGS-B on a single weighted cost) -- least_squares
    is purpose-built for exactly this "drive many residuals to zero"
    problem and converges far more reliably to an exact solution when one
    exists, including near joint-limit boundaries.

    Tries several starting points (multi-start) and returns the best
    result by final residual norm.

    Parameters:
        T_target  : 4x4 target pose
        bounds_rad: list of (min, max) tuples in RADIANS, one per joint
        qh        : fixed helper joint parameter (unchanged, e.g. 0.0)
        n_starts  : number of random starting points to try, in addition
                    to the midpoint of the bounds
        w_pos, w_orient: relative weight of position vs orientation
                    residuals (equal by default -- unlike the old scalar
                    cost, there is no need to underweight orientation)
        primary_guess: optional joint angles (radians) to try FIRST --
                    see docstring note below on redundancy.

    Returns: (q_solved_rad, cost, success)
        cost is the final sum-of-squared-residuals (near 0 = good fit).
        success is True only if the residual norm is below a sane
        threshold -- always check this before trusting q.

    Note on redundancy: a 5-DOF arm can have MULTIPLE joint
    configurations reaching the same pose. The solver has no way to know
    which one you want unless you tell it via primary_guess -- without
    it, you may get a different (but equally valid) solution than the
    one you had in mind.
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
        q0_clipped = np.clip(q0, lo, hi)  # least_squares requires x0 strictly within bounds
        res = least_squares(_pose_residuals, q0_clipped,
                             args=(T_target, qh, w_pos, w_orient),
                             bounds=(lo, hi), method="trf")
        cost = np.sum(res.fun**2)
        if best is None or cost < best[0]:
            best = (cost, res.x)
        if primary_guess is not None and cost < 1e-10:
            break

    cost, q_sol = best
    success = bool(cost < 1e-6)
    return q_sol, cost, success