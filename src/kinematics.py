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
    [1, 0, 0, 87],  # TODO: student lab
    [0, -1, 0, 0],
    [0, 0, -1, 154.2],
    [0, 0, 0, 1],
], dtype=float)

S_list = np.array([
    [0, 0, 0, 0, 0, 0],  # TODO: student lab
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
], dtype=float)


def to_s_matrix(w, v):
    """
    Build the 4x4 [S] skew-symmetric matrix for a screw axis.
    w: (3,) angular velocity component
    v: (3,) linear velocity component
    """
    # TODO: student lab
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
    pass


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
    """
    Inverse kinematics for the Lite 6.

    pose: [x, y, z, phi, theta, psi]  (mm, radians).

    Returns joint angles in radians as a numpy array of length 6,
    or None if the pose is not reachable.
    """
    # TODO: student lab
    return None


# ======================================================================
# IK - Numerical Method (bounded Gauss-Newton)
# ======================================================================

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
    # TODO: student lab
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

        our_pose = np.array(get_pose_from_T(FK_dh(DH_STD, q, 6)))
        qs[i] = q
        pos_err[i] = np.linalg.norm(pose_sdk[:3] - our_pose[:3])

        rot_sdk = rot_matrix(*pose_sdk[3:])
        our_rot = rot_matrix(*our_pose[3:])

        r_err = our_rot.T @ rot_sdk
        or_err[i] = arccos((trace(r_err) - 1) / 2)

    or_err = np.degrees(or_err)
    print(f"Position Error Mean: {np.mean(pos_err)}, Median: {np.median(pos_err)}, Max: {np.max(pos_err)}, 95%: {np.percentile(pos_err, 95)}")
    print(f"Orientation Error Mean: {np.mean(or_err)}, Median: {np.median(or_err)}, Max: {np.max(or_err)}, 95%: {np.percentile(or_err, 95)}")

    import matplotlib.pyplot as plt

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
