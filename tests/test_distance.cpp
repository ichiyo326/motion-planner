#include <gtest/gtest.h>
#include "geometry/distance.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  Known-answer tests for primitive distance kernels.
// ─────────────────────────────────────────────────────────────────────────────

// Capsule along Z axis, centre at origin, half_length=0.1, radius=0.03
static const Vec3 CAP_P(0, 0, -0.1);
static const Vec3 CAP_Q(0, 0,  0.1);
static const double CAP_R = 0.03;

// ── capsule–sphere ────────────────────────────────────────────────────────────

TEST(DistanceCapsuleSphere, Separated_SphereOnAxis) {
    // Sphere centre at (0,0,0.3), radius=0.05
    // Closest point on capsule axis = (0,0,0.1) (endpoint)
    // dist = 0.2, sd = 0.2 - 0.03 - 0.05 = 0.12
    auto r = distRobotCapsuleVsSphere(CAP_P, CAP_Q, CAP_R,
                                      Vec3(0,0,0.3), 0.05);
    EXPECT_NEAR(r.sd, 0.12, 1e-6);
    EXPECT_GT(r.sd, 0.0);
}

TEST(DistanceCapsuleSphere, Separated_SphereLateral) {
    // Sphere centre at (0.2,0,0), radius=0.05
    // Closest point on capsule axis = (0,0,0) (on segment)
    // dist = 0.2, sd = 0.2 - 0.03 - 0.05 = 0.12
    auto r = distRobotCapsuleVsSphere(CAP_P, CAP_Q, CAP_R,
                                      Vec3(0.2,0,0), 0.05);
    EXPECT_NEAR(r.sd, 0.12, 1e-6);
}

TEST(DistanceCapsuleSphere, Penetrating) {
    // Sphere centre at (0.02,0,0), radius=0.05 → overlaps with capsule
    auto r = distRobotCapsuleVsSphere(CAP_P, CAP_Q, CAP_R,
                                      Vec3(0.02,0,0), 0.05);
    EXPECT_LT(r.sd, 0.0);
}

// ── capsule–capsule ───────────────────────────────────────────────────────────

TEST(DistanceCapsuleCapsule, Parallel_Separated) {
    // Robot capsule along Z, obstacle capsule offset laterally
    Vec3 obsP(0.2, 0, -0.1), obsQ(0.2, 0, 0.1);
    double obsR = 0.03;
    auto r = distRobotCapsuleVsCapsule(CAP_P, CAP_Q, CAP_R,
                                       obsP, obsQ, obsR);
    // dist between axes = 0.2, sd = 0.2 - 0.03 - 0.03 = 0.14
    EXPECT_NEAR(r.sd, 0.14, 1e-5);
}

TEST(DistanceCapsuleCapsule, Perpendicular) {
    // Robot: Z axis. Obstacle: X axis, both at origin
    Vec3 obsP(-0.1, 0, 0), obsQ(0.1, 0, 0);
    double obsR = 0.02;
    auto r = distRobotCapsuleVsCapsule(CAP_P, CAP_Q, CAP_R,
                                       obsP, obsQ, obsR);
    // Closest points both at origin, dist=0 → penetrating
    EXPECT_LE(r.sd, 0.0);
}

// ── capsule–box ───────────────────────────────────────────────────────────────

TEST(DistanceCapsuleBox, SeparatedAbove) {
    // Box centred at (0,0,0.5) with half-extent 0.1 in all dims
    // Capsule top endpoint at (0,0,0.1); closest box surface at z=0.4
    // dist = 0.3, sd = 0.3 - 0.03 = 0.27
    Mat3 R = Mat3::Identity();
    Vec3 he(0.1, 0.1, 0.1);
    auto r = distRobotCapsuleVsBox(CAP_P, CAP_Q, CAP_R,
                                   Vec3(0,0,0.5), R, he);
    EXPECT_NEAR(r.sd, 0.27, 1e-4);
    EXPECT_GT(r.sd, 0.0);
}

TEST(DistanceCapsuleBox, Penetrating) {
    // Box overlapping with capsule
    Mat3 R = Mat3::Identity();
    Vec3 he(0.5, 0.5, 0.5);
    auto r = distRobotCapsuleVsBox(CAP_P, CAP_Q, CAP_R,
                                   Vec3(0,0,0), R, he);
    EXPECT_LT(r.sd, 0.0);
}
