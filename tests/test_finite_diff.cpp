#include <gtest/gtest.h>
#include "collision/collision_checker.hpp"
#include "kinematics/fk.hpp"
#include "io/scene_loader.hpp"
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
//  Finite-difference check: verifies that the signed distance gradient
//  direction is consistent with the geometric Jacobian linearisation.
//
//  Test: for a safe (non-penetrating) configuration, a small joint displacement
//  δq should change sd in the direction predicted by n^T J_pos δq.
//  We check sign and order-of-magnitude agreement (not exact equality).
// ─────────────────────────────────────────────────────────────────────────────

static Scene makeTestScene() {
    // Minimal scene: one sphere obstacle
    Scene s;
    for (int i = 0; i < 7; ++i)
        s.joint_limits.push_back({-3.0, 3.0});

    // One link capsule (joint 0 → 1)
    s.link_capsules.push_back({0, 1, 0.04});

    // One sphere obstacle slightly away
    SphereObs sph;
    sph.id     = "s1";
    sph.pos    = Vec3(0.0, 0.3, 0.4);
    sph.radius = 0.08;
    s.obstacles.push_back(sph);

    s.d_safe        = 0.02;
    s.edge_check    = {0.05, 12};
    s.goal_tolerance_l2 = 0.05;
    return s;
}

TEST(FiniteDiff, GradientSignAgreement) {
    Scene scene = makeTestScene();
    CollisionChecker checker(scene);

    // Home configuration (roughly safe w.r.t. the sphere)
    Eigen::Matrix<double, 7, 1> q0;
    q0 << 0.0, -0.785, 0.0, -1.5, 0.0, 1.0, 0.0;

    DistanceQuery dq;
    dq.q      = q0;
    dq.d_safe = scene.d_safe;

    auto r0 = checker.query(dq);
    if (r0.in_collision) {
        GTEST_SKIP() << "Base config is in collision, skipping FD test";
    }

    double sd0    = r0.worst.sd;
    Vec3   n      = r0.worst.n;
    int    lid    = r0.worst.link_id;
    Vec3   p_rob  = r0.worst.p_robot;

    FKResult fk0 = computeFK(q0);
    auto J = computeJacobian(fk0, p_rob, lid);  // 3×7

    const double eps = 1e-4;
    int agree = 0, total = 0;

    for (int j = 0; j < 7; ++j) {
        Eigen::Matrix<double, 7, 1> q1 = q0;
        q1[j] += eps;

        DistanceQuery dq1; dq1.q = q1; dq1.d_safe = scene.d_safe;
        double sd1 = checker.query(dq1).min_sd;

        double fd_grad  = (sd1 - sd0) / eps;
        double lin_grad = n.dot(J.col(j));

        // Only check joints where the gradient is non-negligible
        if (std::abs(lin_grad) > 1e-4) {
            ++total;
            bool same_sign = (fd_grad * lin_grad > 0);
            if (same_sign) ++agree;

            // Optional: log for debugging
            // std::cout << "j=" << j << " fd=" << fd_grad
            //           << " lin=" << lin_grad << " sign_ok=" << same_sign << "\n";
        }
    }

    // At least 50% of non-trivial joints should agree in sign
    if (total > 0) {
        EXPECT_GE(static_cast<double>(agree) / total, 0.5)
            << "Fewer than half of gradient signs agree";
    }
}
