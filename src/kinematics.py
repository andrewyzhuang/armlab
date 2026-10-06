"""
Kinematics for the UFactory Lite 6.

TODO: Implement all functions marked with TODO below.
      For FK, implement either the DH method or the PoX method (your choice).
      For IK, implement IK_geometric.
"""

import numpy as np
from numpy import *
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, Circle


# ======================================================================
# FK - DH Method
# ======================================================================

# Standard DH parameters for the UFactory Lite 6: https://docs.supportarticle.ufactory.cc/support_articles/developer/kinematic-and-dynamic-parameters/lite6.html
# One row per joint: [theta_offset (rad), d (mm), alpha (rad), a (mm)]
DH_STD = np.array([
    [0.0, 243.3, -pi/2, 0.0],
    [-pi/2, 0.0, pi, 200],
    [-pi/2, 0.0, pi/2, 87],
    [0.0, 227.6, pi/2, 0.0],
    [0.0, 0.0, -pi/2, 0.0],
    [0.0, 61.5, 0.0, 0.0],
], dtype=float)

# Lite 6 joint travel limits, matching the MuJoCo model
# (robot_assets/UfactoryLite6/xml/lite6.xml). Joint 3 is asymmetric.
JOINT_LIMITS_DEG = np.array([
    [-360.0, 360.0],   # Joint 1
    [-150.0, 150.0],   # Joint 2
    [  -3.5, 300.0],   # Joint 3
    [-360.0, 360.0],   # Joint 4
    [-124.0, 124.0],   # Joint 5
    [-360.0, 360.0],   # Joint 6
], dtype=float)
JOINT_LIMITS = np.radians(JOINT_LIMITS_DEG)   # (6, 2) in radians

# The arm's default (home) configuration, matching the UFactory Studio default.
# This is both the pose Initial Pose drives to and the seed IK_numerical starts from.
Q_DEFAULT_DEG = np.array([0.0, 9.9, 31.8, 0.0, 21.9, 0.0], dtype=float)
Q_DEFAULT = np.radians(Q_DEFAULT_DEG)         # (6,) in radians


def get_transform_from_dh(theta_offset, d, alpha, a, joint_angle):
    """
    Build the 4x4 transform for one DH row.

    DH row format (from arm.dh_params):
        theta_offset (rad), d (mm), alpha (rad), a (mm)
    joint_angle: current joint angle in radians.
    """

    theta = theta_offset + joint_angle

    T = np.array([[cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)],
         [sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
         [0, sin(alpha), cos(alpha), d],
         [0, 0, 0, 1]], dtype=float)
    
    return T


def FK_dh(dh_params, joint_angles_rad, num_joints):
    """
    Forward kinematics via DH convention.

    Calls get_transform_from_dh for each joint and chains the results.

    dh_params:        DH table, flat list (SDK arm.dh_params has 7 rows, 4 values) or a (n, 4) array.
                      Each row: [theta_offset (rad), d (mm), alpha (rad), a (mm)]
    joint_angles_rad: joint angles in radians, length num_joints.
    num_joints:       number of joints to chain (pass 6 for end-effector).

    Returns 4x4 homogeneous transform (base -> end-effector).
    """
    M = np.eye(4)

    for i in range(num_joints):

        theta_off, d, alpha, a = dh_params[i]
        M = M @ get_transform_from_dh(theta_off, d, alpha, a, joint_angles_rad[i])

    return M



# ======================================================================
# FK - PoX Method
# ======================================================================

# M and S_list are robot constants for the PoX method:
#   M:      4x4 end-effector transform at the zero (home) configuration.
#   S_list: 6x6 - one space-frame screw axis per row: [w1, w2, w3, v1, v2, v3]
# Units are mm, like FK_dh.

M = np.array([
    [1, 0, 0, 87],
    [0, -1, 0, 0],
    [0, 0, -1, 154.2],
    [0, 0, 0, 1],
], dtype=float)

S_list = np.array([
    [0, 0, 1, 0, 0, 0],
    [0, 1, 0, -243.3, 0, 0],
    [0, -1, 0, 443.3, 0, 0],
    [0, 0, -1, 0, 87, 0],
    [0, 1, 0, -215.7, 0, 87],
    [0, 0, -1, 0, 87, 0],
], dtype=float)


def to_s_matrix(w, v):
    """
    Build the 4x4 [S] skew-symmetric matrix for a screw axis.
    w: (3,) angular velocity component
    v: (3,) linear velocity component
    """
    # TODO: student lab

    omega_hat = np.array([[0 , -1 * w[2], w[1]], 
                      [w[2], 0, -1 * w[0]], 
                      [-1 * w[1], w[0], 0],
                      ] , dtype=float )

    s = np.zeros((4,4))
    s[:3,:3]  = omega_hat
    s[:3, 3] = v
    return s
    pass


def FK_pox(joint_angles_rad, m_mat, s_lst):
    """
    Forward kinematics via product of exponentials.

    Calls to_s_matrix for each joint and chains e^([S]*theta).

    joint_angles_rad: joint angles in radians (raw angles, offsets live in M).
    m_mat:            4x4 home configuration matrix M.
    s_lst:            (6, 6) screw axes, one per row: [w1, w2, w3, v1, v2, v3].

    Returns 4x4 homogeneous transform (base -> end-effector).
    """
    # TODO: student lab
    T = np.array([
    [1, 0, 0, 0],  # TODO: student lab
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    ], dtype=float)

    w = s_lst[:, :3]
    v = s_lst[:, -3:]

    for i in range(len(joint_angles_rad)):
        joint_w = w[i, :]
        joint_v = v[i, :]
        s = to_s_matrix(joint_w, joint_v)
        T = T @ expm(s * joint_angles_rad[i])

    T = T @ m_mat

    return T


# ======================================================================
# FK - shared output helper
# ======================================================================

def get_pose_from_T(T):
    """
    Extract [x, y, z, phi, theta, psi] from a 4x4 homogeneous transform.
    x, y, z  in mm.  phi, theta, psi  in radians.

    Orientation is roll-pitch-yaw about fixed X, Y, Z axes (matches the
    xArm SDK convention and the GUI's Roll/Pitch/Yaw readout).

    Used by both FK_dh and FK_pox to produce the pose vector for the GUI.
    """
    phi = float(atan2(T[2, 1], T[2, 2]))
    theta = float(atan2(-T[2, 0], sqrt(T[0, 0]**2 + T[1, 0]**2)))
    psi = float(atan2(T[1, 0], T[0, 0]))
    x, y, z = T[:3, 3]
    
    return [x, y, z, phi, theta, psi]

# ======================================================================
# IK - Geometric Method
# ======================================================================

def IK_geometric(dh_params, pose):
    """Closed-form IK for the Lite 6 (spherical wrist)."""
    x, y, z, phi, theta, psi_rpy = pose
    R_target = rot_matrix(phi, theta, psi_rpy)

    # --- DH constants read from the table (never hard-code) ---
    d1          = dh_params[0, 1]
    a2          = dh_params[1, 3]
    theta_off_2 = dh_params[1, 0]
    alpha_2     = dh_params[1, 2]
    a3          = dh_params[2, 3]
    theta_off_3 = dh_params[2, 0]
    alpha_3     = dh_params[2, 2]
    d4          = dh_params[3, 1]
    d6          = dh_params[5, 1]

    # --- 1. wrist center ---
    p_wc = np.array([x, y, z], dtype=float) - d6 * R_target[:, 2]

    # --- 2. joint 1 ---
    r = float(np.hypot(p_wc[0], p_wc[1]))
    if r < 1e-6:
        theta1 = 0.0
    else:
        theta1 = float(np.arctan2(p_wc[1], p_wc[0]))

    # --- 3. planar 2-link for joints 2,3 ---
    s = p_wc[2] - d1
    L = float(np.hypot(r, s))
    L2 = float(np.hypot(a3, d4))
    psi_link = float(np.arctan2(d4, a3))

    if L > a2 + L2 or L < abs(a2 - L2) or L < 1e-6:
        return None

    cos_gamma = (a2 * a2 + L2 * L2 - L * L) / (2.0 * a2 * L2)
    cos_beta  = (a2 * a2 + L * L   - L2 * L2) / (2.0 * a2 * L)
    gamma = float(np.arccos(np.clip(cos_gamma, -1.0, 1.0)))
    beta  = float(np.arccos(np.clip(cos_beta,  -1.0, 1.0)))

    alpha = float(np.arctan2(r, s))
    theta2 = alpha - beta
    theta3 = gamma + psi_link - np.pi / 2.0

    # --- 4. wrist orientation ---
    T_01 = get_transform_from_dh(0.0,           d1, -np.pi / 2.0, 0.0,  theta1)
    T_12 = get_transform_from_dh(theta_off_2,   0.0, np.pi,        a2,   theta2)
    T_23 = get_transform_from_dh(theta_off_3,   0.0, np.pi / 2.0,  a3,   theta3)
    R_03 = (T_01 @ T_12 @ T_23)[:3, :3]

    R_36 = R_03.T @ R_target

    c5 = float(np.clip(R_36[2, 2], -1.0, 1.0))
    theta5 = float(np.arccos(c5))
    s5 = float(np.sin(theta5))

    if abs(s5) < 1e-6:
        # wrist singularity: joints 4 and 6 are coupled, pick theta4 = 0
        theta4 = 0.0
        if c5 > 0.0:
            theta6 = float(np.arctan2( R_36[1, 0],  R_36[0, 0]))
        else:
            theta6 = float(np.arctan2( R_36[1, 0], -R_36[0, 0]))
    else:
        theta4 = float(np.arctan2(-R_36[1, 2], -R_36[0, 2]))
        theta6 = float(np.arctan2(-R_36[2, 1],  R_36[2, 0]))

    result = np.array([theta1, theta2, theta3, theta4, theta5, theta6])

    # --- 5. joint limits ---
    if not np.all((result >= JOINT_LIMITS[:, 0]) & (result <= JOINT_LIMITS[:, 1])):
        return None

    return result


# ======================================================================
# IK - Numerical Method (bounded Gauss-Newton)
# ======================================================================

def compute_jacobian(dh_params, q, residual_func, base_res, eps=1e-6):
    """Computes the 6x6 numerical Jacobian using finite differences."""
    J = np.zeros((6, 6))
    for i in range(6):
        q_delta = q.copy()
        q_delta[i] += eps
        res_delta = residual_func(q_delta)
        # Forward finite difference
        J[:, i] = (res_delta - base_res) / eps
    return J


def IK_numerical(dh_params, pose, q0=None, joint_limits=None, w_rot=200.0):
    """
    Inverse kinematics for the Lite 6 by bounded Gauss-Newton least squares.

    Solves  min_q ||W e(q)||^2  s.t.  lb <= q <= ub, where e is the pose error
    between FK_dh(q) and the requested pose.

    pose:         [x, y, z, roll, pitch, yaw]  (mm, radians) - same as IK_geometric.
    q0:           starting guess in radians. None seeds from Q_DEFAULT (the home
                  pose) rather than the arm's current angles, so repeated calls
                  are reproducible.
    joint_limits: (6, 2) array of [lo, hi] in radians; defaults to JOINT_LIMITS.
    w_rot:        mm per radian, putting the position and rotation blocks of the
                  residual on one scale. Only matters when a limit binds and the
                  pose cannot be reached exactly.

    Returns joint angles in radians as a numpy array of length 6,
    or None if the pose is not reachable.
    """
    
    if q0 is None:
        q0 = Q_DEFAULT
    q = np.array(q0, dtype=float).copy()
    
    if joint_limits is None:
        joint_limits = JOINT_LIMITS
    lo = joint_limits[:, 0]
    hi = joint_limits[:, 1]
    
    # Custom optimization parameters
    max_steps = 150
    pos_tol = 0.1       # 0.1 mm position accuracy limit
    rot_tol = 0.005     # 0.005 rad orientation accuracy limit
    damping = 1e-4      # Levenberg-Marquardt damping factor to pass wrist singularities

    x, y, z, phi, theta, psi_rpy = pose
    R_target = rot_matrix(phi, theta, psi_rpy)
    p_target = np.array([x, y, z], dtype=float)

    def calculate_residual(q_curr):
        "Finding the position error the orientational offset"
        T = FK_dh(dh_params, q_curr, 6)
        p = T[:3, 3]
        R = T[:3, :3]

        e_pos = p - p_target

        # Rotation vector of (R_target · R^T): the world-frame rotation
        # needed to take the FK orientation to the target orientation.
        R_err = R_target @ R.T
        v = np.array([R_err[2, 1] - R_err[1, 2],
                      R_err[0, 2] - R_err[2, 0],
                      R_err[1, 0] - R_err[0, 1]])
        n = float(np.linalg.norm(v))
        
        if n < 1e-12:
            omega = np.zeros(3)
        else:
            angle = float(np.arctan2(n / 2.0, (np.trace(R_err) - 1.0) / 2.0))
            omega = v / n * angle

        return np.concatenate([e_pos, w_rot * omega])

    # Initial seeding clip to enforce starting bounds sanity
    q = np.clip(q, lo, hi)

    for step in range(max_steps):
        res = calculate_residual(q)
        
        # Pull separate physical metrics for convergence checks
        e_pos_norm = np.linalg.norm(res[:3])
        e_rot_norm = np.linalg.norm(res[3:]) / w_rot

        # Check if the current joint state meets the assignment tolerances
        if e_pos_norm <= pos_tol and e_rot_norm <= rot_tol:
            return q

        # Compute the 6x6 numerical Jacobian matrix
        J = compute_jacobian(dh_params, q, calculate_residual, res)

        # Solve damped normal equations (LM step update): (J^T*J + damping*I)*dq = -J^T*res
        A = J.T @ J + damping * np.eye(6)
        b = -J.T @ res
        dq = np.linalg.solve(A, b)

        # Step forward, then force boundary constraints via clipping
        q += dq
        q = np.clip(q, lo, hi)

    # Final post-loop evaluation check 
    res = calculate_residual(q)
    if np.linalg.norm(res[:3]) <= pos_tol and (np.linalg.norm(res[3:]) / w_rot) <= rot_tol:
        return q

    # Return None if the target is unreachable or failed to converge within max steps
    return None


def rot_matrix(phi, theta, psi):
    """
    Build the 3x3 rotation matrix R = Rz(psi) @ Ry(theta) @ Rx(phi),
    the fixed-angle (roll-pitch-yaw about fixed X, Y, Z) convention used
    by get_pose_from_T.
    """
    return np.array([
        [cos(psi)*cos(theta), cos(psi)*sin(theta)*sin(phi) - sin(psi)*cos(phi), cos(psi)*sin(theta)*cos(phi) + sin(psi)*sin(phi)],
        [sin(psi)*cos(theta), sin(psi)*sin(theta)*sin(phi) + cos(psi)*cos(phi), sin(psi)*sin(theta)*cos(phi) - cos(psi)*sin(phi)],
        [-sin(theta), cos(theta)*sin(phi), cos(theta)*cos(phi)],
    ], dtype=float)

def error_test(arm, iterations=100):

    lower = JOINT_LIMITS_DEG[:, 0]
    upper = JOINT_LIMITS_DEG[:, 1]
    qs = np.zeros((iterations, 6))
    pos_err = np.zeros(iterations)
    or_err = np.zeros(iterations)

    for i in range(iterations):

        q = np.random.uniform(low=lower, high=upper)

        code, pose_sdk = arm.xarm.get_forward_kinematics(q)
        if code != 0:
            continue
        pose_sdk = np.array(pose_sdk)

        our_pose = np.array(get_pose_from_T(FK_pox(q, M, S_list)))
        qs[i] = q
        pos_err[i] = np.linalg.norm(pose_sdk[:3] - our_pose[:3])

        rot_sdk = rot_matrix(*pose_sdk[3:])
        our_rot = rot_matrix(*our_pose[3:])

        r_err = our_rot.T @ rot_sdk
        or_err[i] = arccos((trace(r_err) - 1) / 2)

    or_err = np.degrees(or_err)
    print(f"Position Error Mean: {np.mean(pos_err)}, Median: {np.median(pos_err)}, Max: {np.max(pos_err)}, 95%: {np.percentile(pos_err, 95)}")
    print(f"Orientation Error Mean: {np.mean(or_err)}, Median: {np.median(or_err)}, Max: {np.max(or_err)}, 95%: {np.percentile(or_err, 95)}")


    plt.figure()
    plt.hist(pos_err, bins=30)
    plt.xlabel("Position error (mm)")
    plt.ylabel("Count")
    plt.title("Position error histogram")

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for j, ax in enumerate(axes.flat):
        ax.scatter(qs[:, j], pos_err, s=10)
        ax.set_xlabel(f"Joint {j+1} (deg)")
        ax.set_ylabel("Position error (mm)")
    fig.tight_layout()

    plt.show()


def _ik_pose_error(T_target, q):
    """Position (mm) and orientation (deg) error between T_target and FK(q)."""
    T = FK_dh(DH_STD, q, 6)
    p_err = np.linalg.norm(T[:3, 3] - T_target[:3, 3])
    R_err = T_target[:3, :3].T @ T[:3, :3]
    o_err = np.degrees(arccos(np.clip((trace(R_err) - 1) / 2, -1.0, 1.0)))
    return p_err, o_err


def _ik_numerical_lsq(pose, q0=Q_DEFAULT, w_rot=200.0):
    """Bounded least squares on [position error, w_rot * rotation-vector error], seeded from home."""

    lower, upper = JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
    p_t = np.asarray(pose[:3], dtype=float)
    R_t = rot_matrix(*pose[3:])

    def residual(q):
        T = FK_dh(DH_STD, q, 6)
        rv = Rotation.from_matrix(R_t.T @ T[:3, :3]).as_rotvec()
        return np.concatenate([T[:3, 3] - p_t, w_rot * rv])

    q0 = np.clip(q0, lower + 1e-6, upper - 1e-6)
    sol = least_squares(residual, q0, bounds=(lower, upper), xtol=1e-12, ftol=1e-12, gtol=1e-12)
    T_t = np.eye(4)
    T_t[:3, :3], T_t[:3, 3] = R_t, p_t
    p_err, o_err = _ik_pose_error(T_t, sol.x)
    if p_err > 0.1 or o_err > 0.01:
        return None
    return sol.x


def _ik_round_trip(solvers, iterations, seed, title, separate=False):
    """
    Draw random q inside JOINT_LIMITS, FK -> IK -> FK through each solver, and
    compare poses (not angles: a different branch reaching the same pose is fine).
    Prints mean/max position and orientation error per solver and plots histograms:
    overlaid on one figure, or with separate=True one subplot per solver per error type.
    """

    rng = np.random.default_rng(seed)
    qs = rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1], size=(iterations, 6))
    results = {name: {"pos": [], "ori": [], "fail": 0} for name in solvers}

    for q in qs:
        T_target = FK_dh(DH_STD, q, 6)
        pose = get_pose_from_T(T_target)
        for name, solve in solvers.items():
            q_ik = solve(pose)
            if q_ik is None:
                results[name]["fail"] += 1
                continue
            p_err, o_err = _ik_pose_error(T_target, q_ik)
            results[name]["pos"].append(p_err)
            results[name]["ori"].append(o_err)

    print(f"\n{title}: {iterations} random joint vectors (FK -> IK -> FK, pose compared)")
    print(f"{'solver':<14}{'solved':>8}{'pos mean':>12}{'pos max':>12}{'ori mean':>12}{'ori max':>12}")
    for name, r in results.items():
        pos, ori = np.array(r["pos"]), np.array(r["ori"])
        if len(pos) == 0:
            print(f"{name:<14}{0:>8}  (no solutions)")
            continue
        print(f"{name:<14}{len(pos):>8}{pos.mean():>10.2e}mm{pos.max():>10.2e}mm"
              f"{ori.mean():>9.2e}deg{ori.max():>9.2e}deg")

    if separate:
        # one subplot per solver (rows) per error type (columns)
        metrics = (("pos", "Position error (mm)"), ("ori", "Orientation error (deg)"))
        fig, axes = plt.subplots(len(results), 2, figsize=(11, 3.5 * len(results)), squeeze=False)
        for row, (name, r) in zip(axes, results.items()):
            for ax, (key, label) in zip(row, metrics):
                if r[key]:
                    ax.hist(r[key], bins=30)
                ax.set_xlabel(label)
                ax.set_ylabel("Count")
                ax.set_title(f"{name}: {label.split(' (')[0].lower()}")
        fig.suptitle(title)
        fig.tight_layout()
        return results

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, r in results.items():
        if r["pos"]:
            axes[0].hist(r["pos"], bins=30, alpha=0.6, label=name)
            axes[1].hist(r["ori"], bins=30, alpha=0.6, label=name)
    axes[0].set_xlabel("Position error (mm)")
    axes[1].set_xlabel("Orientation error (deg)")
    for ax in axes:
        ax.set_ylabel("Count")
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()

    return results


def error_test_ik(iterations=300, seed=0):
    """
    IK round trip inside our own model: IK_geometric vs the numerical solver.
    No hardware needed, and no error floor - both should come back exact.
    """

    solvers = {
        "IK_geometric": lambda pose: IK_geometric(DH_STD, pose),
        "IK_numerical": _ik_numerical_lsq,
    }
    results = _ik_round_trip(solvers, iterations, seed, "IK round trip (geometric vs numerical)",
                             separate=True)
    plt.show()
    return results


def error_test_ik_vendor(arm, iterations=300, seed=0):
    """
    Same sweep through the vendor's arm.xarm.get_inverse_kinematics, checked
    with our FK. Needs the real arm, but the arm never moves. The vendor uses a
    per-arm factory calibration our model lacks, so allow up to 10 mm.
    """

    def ik_vendor(pose):
        code, q = arm.xarm.get_inverse_kinematics(list(pose))
        return np.array(q[:6], dtype=float) if code == 0 else None

    solvers = {
            "IK_numerical": _ik_numerical_lsq,
            "vendor (SDK)": ik_vendor
        }
    results = _ik_round_trip(solvers, iterations, seed, "IK vs vendor", separate=True)
    pos = np.array(results["vendor (SDK)"]["pos"])
    if len(pos):
        print(f"vendor: {np.mean(pos <= 10.0) * 100:.1f}% of solved poses within the 10 mm allowance")
    plt.show()
    return results


def reachability_map(blocks=(38.0, 25.0), step=10.0, flange_above_center=85.0):
    """
    Top-down grasp reachability over the whole board using IK_geometric.

    For each block size the flange sits flange_above_center mm above the block's
    center, tool straight down (roll = pi, pitch = 0), and the gripper is tried
    aligned to the board grid (yaw = 0) and turned 90 deg (yaw = pi/2). A point
    is reachable if IK_geometric returns a solution inside the joint limits.
    """

    # Board extents, matching camera.py (MIN_X..MAX_X, MIN_Y..MAX_Y).
    board_x, board_y = (-50.0, 450.0), (-450.0, 450.0)
    xs = np.arange(board_x[0], board_x[1] + 1e-9, step)
    ys = np.arange(board_y[0], board_y[1] + 1e-9, step)

    maps = {}
    for block in blocks:
        z = block / 2.0 + flange_above_center
        reach = np.zeros((len(ys), len(xs), 2), dtype=bool)   # [..., 0] aligned, [..., 1] turned 90 deg
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                for k, yaw in enumerate((0.0, pi / 2)):
                    reach[i, j, k] = IK_geometric(DH_STD, [x, y, z, pi, 0.0, yaw]) is not None
        maps[block] = reach
        print(f"\n{block:.0f} mm block (flange z = {z:.1f} mm), {reach.shape[0] * reach.shape[1]} grid points:")
        print(f"  aligned: {reach[..., 0].sum()}  turned 90: {reach[..., 1].sum()}  "
              f"both: {(reach[..., 0] & reach[..., 1]).sum()}  either: {(reach[..., 0] | reach[..., 1]).sum()}")

    if len(blocks) == 2:
        diff = (maps[blocks[0]] != maps[blocks[1]]).any(axis=2).sum()
        print(f"\nGrid points whose reachability differs between {blocks[0]:.0f} mm and {blocks[1]:.0f} mm: {diff}")

    cmap = ListedColormap(["#d9d9d9", "#f4a259", "#5b8e7d"])   # neither, one orientation, both
    fig, axes = plt.subplots(1, len(blocks), figsize=(6.5 * len(blocks), 6.5), sharey=True, squeeze=False)
    for ax, (block, reach) in zip(axes[0], maps.items()):
        ax.imshow(reach.sum(axis=2), origin="lower", cmap=cmap, vmin=0, vmax=2,
                  extent=(xs[0] - step / 2, xs[-1] + step / 2, ys[0] - step / 2, ys[-1] + step / 2))
        ax.add_patch(Rectangle((board_x[0], board_y[0]), board_x[1] - board_x[0], board_y[1] - board_y[0],
                               fill=False, lw=2, ec="k"))
        ax.add_patch(Circle((0, 0), 40, fc="k", ec="k"))
        ax.annotate("base", (0, 0), xytext=(15, 50), textcoords="offset points")
        ax.set_title(f"{block:.0f} mm block, flange z = {block / 2 + flange_above_center:.1f} mm")
        ax.set_xlabel("x (mm)")
        ax.set_aspect("equal")
    axes[0, 0].set_ylabel("y (mm)")
    handles = [Rectangle((0, 0), 1, 1, fc=c) for c in cmap.colors]
    fig.legend(handles, ["unreachable", "one orientation only", "both orientations"],
               loc="lower center", ncol=3)
    fig.suptitle("Top-down grasp reachability (IK_geometric, within joint limits)")
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    plt.show()
    return maps


# ======================================================================
# Standalone test
# ======================================================================

if __name__ == '__main__':
    # get_transform_from_dh with all zeros should return identity

    T = get_transform_from_dh(0, 0, 0, 0, 0)

    if T is not None:
        print("get_transform_from_dh(all zeros):")
        print(T)
    else:
        print("get_transform_from_dh not yet implemented")
