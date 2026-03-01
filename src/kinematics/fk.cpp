#include "fk.hpp"
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
//  Simplified Panda-like DH parameters (approximate geometry only).
//  Modified DH: [a, d, alpha, theta_offset]
//
//  Reference: Franka Emika Panda — values approximated for demo.
// ─────────────────────────────────────────────────────────────────────────────

struct DHParams { double a, d, alpha; };

static const DHParams DH[7] = {
    { 0.000,  0.333,  0.0      },  // Joint 1
    { 0.000,  0.000, -M_PI/2.0 },  // Joint 2
    { 0.000,  0.316,  M_PI/2.0 },  // Joint 3
    { 0.0825, 0.000,  M_PI/2.0 },  // Joint 4
    {-0.0825, 0.384, -M_PI/2.0 },  // Joint 5
    { 0.000,  0.000,  M_PI/2.0 },  // Joint 6
    { 0.088,  0.000,  M_PI/2.0 }   // Joint 7
};

// Build a 4×4 homogeneous DH transform as (pos, rot) pair
static std::pair<Vec3, Quat> dhTransform(double a, double d, double alpha, double theta) {
    double ca = std::cos(alpha), sa = std::sin(alpha);
    double ct = std::cos(theta), st = std::sin(theta);

    // Rotation: Rz(theta) * Rx(alpha)
    Mat3 R;
    R << ct, -st*ca,  st*sa,
         st,  ct*ca, -ct*sa,
          0,  sa,     ca;

    Vec3 t(a*ct, a*st, d);
    return { t, Quat(R) };
}

FKResult computeFK(const Eigen::Matrix<double, 7, 1>& q) {
    FKResult frames(8);
    // Frame 0 = world base
    frames[0].pos = Vec3::Zero();
    frames[0].rot = Quat::Identity();

    for (int i = 0; i < 7; ++i) {
        auto [t_local, r_local] = dhTransform(DH[i].a, DH[i].d, DH[i].alpha, q[i]);
        frames[i+1].rot = frames[i].rot * r_local;
        frames[i+1].pos = frames[i].pos + frames[i].rot * t_local;
    }
    return frames;
}

// ── Geometric Jacobian (position part only, 3×7) ─────────────────────────────
Eigen::Matrix<double, 3, 7> computeJacobian(
        const FKResult& fk,
        const Vec3& p_world,
        int link_id) {
    // link_id = capsule index → joint_j = link_id+1
    int tip_joint = std::min(link_id + 1, 7);

    Eigen::Matrix<double, 3, 7> J = Eigen::Matrix<double, 3, 7>::Zero();
    for (int i = 0; i < tip_joint; ++i) {
        // Z-axis of joint i in world frame
        Vec3 z = fk[i].rot * Vec3(0, 0, 1);
        Vec3 r = p_world - fk[i].pos;
        J.col(i) = z.cross(r);
    }
    return J;
}
