#include <gtest/gtest.h>
#include "geometry/distance.hpp"
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
//  Invariant tests for signed distance, normal direction, and witness points.
//  These MUST pass before TrajOpt is connected.
// ─────────────────────────────────────────────────────────────────────────────

static const Vec3 CAP_P(0, 0, -0.1);
static const Vec3 CAP_Q(0, 0,  0.1);
static const double CAP_R = 0.03;

template<typename Fn>
void checkInvariants(Fn get_result, const std::string& label) {
    DistResult r = get_result();

    // 1. Normal is a unit vector
    EXPECT_NEAR(r.n.norm(), 1.0, 1e-9) << label << ": n is not unit";

    // 2. sd ≈ (p_robot - p_obs) · n
    double reconstructed = (r.p_b - r.p_a).dot(r.n);
    EXPECT_NEAR(r.sd, reconstructed, 1e-5) << label << ": sd != (p_b-p_a)·n";

    // 3. For separated case, n points obstacle → robot (p_b is further in n direction)
    if (r.sd > 0) {
        EXPECT_GT((r.p_b - r.p_a).dot(r.n), -1e-9)
            << label << ": n does not point obstacle→robot";
    }
}

TEST(Normals, CapsuleVsSphere_Lateral) {
    checkInvariants([]{
        return distRobotCapsuleVsSphere(CAP_P, CAP_Q, CAP_R,
                                        Vec3(0.2, 0, 0), 0.05);
    }, "capsule-sphere lateral");
}

TEST(Normals, CapsuleVsSphere_Axial) {
    checkInvariants([]{
        return distRobotCapsuleVsSphere(CAP_P, CAP_Q, CAP_R,
                                        Vec3(0, 0, 0.3), 0.05);
    }, "capsule-sphere axial");
}

TEST(Normals, CapsuleVsCapsule_Parallel) {
    checkInvariants([]{
        return distRobotCapsuleVsCapsule(CAP_P, CAP_Q, CAP_R,
                                         Vec3(0.2,0,-0.1), Vec3(0.2,0,0.1), 0.03);
    }, "capsule-capsule parallel");
}

TEST(Normals, CapsuleVsBox_Above) {
    checkInvariants([]{
        return distRobotCapsuleVsBox(CAP_P, CAP_Q, CAP_R,
                                     Vec3(0,0,0.5), Mat3::Identity(), Vec3(0.1,0.1,0.1));
    }, "capsule-box above");
}

TEST(Normals, SignIsConsistent_Separated) {
    auto r = distRobotCapsuleVsSphere(CAP_P, CAP_Q, CAP_R,
                                      Vec3(0.5, 0, 0), 0.05);
    EXPECT_GT(r.sd, 0.0) << "Expected positive sd for separated pair";
}

TEST(Normals, SignIsConsistent_Penetrating) {
    // Very close sphere (inside capsule radius)
    auto r = distRobotCapsuleVsSphere(CAP_P, CAP_Q, CAP_R,
                                      Vec3(0.01, 0, 0), 0.05);
    EXPECT_LT(r.sd, 0.0) << "Expected negative sd for penetrating pair";
}
