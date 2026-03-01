#pragma once
#include <Eigen/Geometry>
#include "vec3.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  Quaternion convention: [w, x, y, z]  (wxyz order, active rotation)
//  Eigen stores internally as [x,y,z,w] — always construct via (w,x,y,z).
// ─────────────────────────────────────────────────────────────────────────────

using Quat = Eigen::Quaterniond;

/// Make a quaternion from a [w, x, y, z] array (JSON / scene.json order).
inline Quat quatFromWXYZ(double w, double x, double y, double z) {
    return Quat(w, x, y, z);   // Eigen Quaterniond(w,x,y,z)
}

/// Rotate a vector: p_world = R(q) * p_local
inline Vec3 rotate(const Quat& q, const Vec3& v) {
    return q * v;
}

/// Transform a point: p_world = R(q) * p_local + t
inline Vec3 transform(const Quat& q, const Vec3& t, const Vec3& p_local) {
    return q * p_local + t;
}

/// Rotation matrix from quaternion
inline Mat3 toMatrix(const Quat& q) {
    return q.toRotationMatrix();
}
