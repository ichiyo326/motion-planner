#pragma once
#include <vector>
#include <Eigen/Core>
#include "math/quat.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  Forward Kinematics for a Panda-like 7-DoF arm.
//
//  DH-inspired simplified model.  Joint frames are defined such that:
//    joint_frame[k].translation()  = position of joint k in world frame
//
//  NOTE: This is NOT a strict Franka Panda implementation.
//        The geometry approximates the Panda for demonstration purposes.
// ─────────────────────────────────────────────────────────────────────────────

struct JointFrame {
    Vec3 pos;   ///< position in world frame
    Quat rot;   ///< orientation in world frame
};

using FKResult = std::vector<JointFrame>;  // index 0..7 (8 frames for 7 joints)

/// Compute FK for a 7-DoF Panda-like arm.
/// Returns 8 joint frames (index 0 = base, 7 = end-effector attachment).
FKResult computeFK(const Eigen::Matrix<double, 7, 1>& q);

/// Jacobian (3×7) for the position of a point p_local on link link_id.
/// link_id corresponds to the capsule endpoint joint_j.
Eigen::Matrix<double, 3, 7> computeJacobian(
    const FKResult& fk,
    const Vec3& p_world,   ///< point in world frame (e.g. witness point)
    int link_id            ///< which link (0-based capsule index)
);
