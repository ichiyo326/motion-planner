#pragma once
#include <cmath>
#include <algorithm>
#include "math/vec3.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  Primitive distance kernels.
//  All functions return:
//    sd        — signed distance (>0 separated, <0 penetrating)
//    p_a, p_b  — witness points on shape A and shape B respectively
//    n         — unit normal pointing A → B  (push B away from A)
//
//  In CollisionChecker:  A = obstacle,  B = robot capsule
//  So n = obstacle → robot  (Design.md §4.1)
// ─────────────────────────────────────────────────────────────────────────────

struct DistResult {
    double sd{};
    Vec3   p_a{};   ///< closest point on obstacle
    Vec3   p_b{};   ///< closest point on robot capsule
    Vec3   n{};     ///< unit normal: obstacle → robot
};

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Closest point on segment [p, q] to point x; returns parameter t ∈ [0,1].
inline Vec3 closestPointOnSegment(const Vec3& p, const Vec3& q, const Vec3& x,
                                  double* t_out = nullptr) {
    Vec3 d = q - p;
    double len2 = d.squaredNorm();
    if (len2 < 1e-12) {
        if (t_out) *t_out = 0.0;
        return p;
    }
    double t = std::clamp((x - p).dot(d) / len2, 0.0, 1.0);
    if (t_out) *t_out = t;
    return p + t * d;
}

/// Closest points between two segments [p1,q1] and [p2,q2].
inline void closestPointsSegSeg(const Vec3& p1, const Vec3& q1,
                                 const Vec3& p2, const Vec3& q2,
                                 Vec3& c1, Vec3& c2) {
    Vec3  d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
    double a = d1.squaredNorm(), e = d2.squaredNorm();
    double f = d2.dot(r);
    double s, t;

    if (a < 1e-12 && e < 1e-12) { c1 = p1; c2 = p2; return; }
    if (a < 1e-12) {
        s = 0.0; t = std::clamp(f / e, 0.0, 1.0);
    } else {
        double c = d1.dot(r);
        if (e < 1e-12) {
            t = 0.0; s = std::clamp(-c / a, 0.0, 1.0);
        } else {
            double b = d1.dot(d2);
            double denom = a * e - b * b;
            s = (denom > 1e-12) ? std::clamp((b * f - c * e) / denom, 0.0, 1.0) : 0.0;
            t = (b * s + f) / e;
            if (t < 0.0) { t = 0.0; s = std::clamp(-c / a, 0.0, 1.0); }
            else if (t > 1.0) { t = 1.0; s = std::clamp((b - c) / a, 0.0, 1.0); }
        }
    }
    c1 = p1 + s * d1;
    c2 = p2 + t * d2;
}

/// Closest point on OBB to point p (OBB defined by center, axes R, half-extents h).
inline Vec3 closestPointOnOBB(const Vec3& center, const Mat3& R,
                               const Vec3& half_extents, const Vec3& p) {
    Vec3 d = p - center;
    Vec3 closest = center;
    for (int i = 0; i < 3; ++i) {
        Vec3 axis = R.col(i);
        double dist = std::clamp(d.dot(axis), -half_extents[i], half_extents[i]);
        closest += dist * axis;
    }
    return closest;
}

// ── Public kernels ───────────────────────────────────────────────────────────

/// Capsule A vs Sphere B
/// capA_p, capA_q — endpoints of capsule A axis; rA — capsule radius
/// sph_c — sphere centre; rB — sphere radius
inline DistResult distCapsuleSphere(
        const Vec3& capA_p, const Vec3& capA_q, double rA,
        const Vec3& sph_c,  double rB) {
    Vec3   pa = closestPointOnSegment(capA_p, capA_q, sph_c);
    Vec3   delta = sph_c - pa;
    double dist  = delta.norm();
    double sd    = dist - rA - rB;

    Vec3 n;
    if (dist < 1e-9) {
        // Degenerate: sphere centre on capsule axis — push outward arbitrarily
        n = Vec3(0, 0, 1);
    } else {
        n = delta / dist;   // capsule → sphere
    }
    // Convention: obstacle(capsule A) → robot(sphere B)
    // Here A=obstacle, so n already points A→B:
    DistResult r;
    r.sd  = sd;
    r.p_a = pa + rA * n;    // point on capsule A surface
    r.p_b = sph_c - rB * n; // point on sphere B surface
    r.n   = n;
    return r;
}

/// Robot Capsule vs Sphere Obstacle
/// (swapped argument order — n is remapped to obstacle→robot)
inline DistResult distRobotCapsuleVsSphere(
        const Vec3& robP, const Vec3& robQ, double rRob,
        const Vec3& sphC, double rSph) {
    // Closest point on robot axis to sphere centre
    Vec3   closest = closestPointOnSegment(robP, robQ, sphC);
    Vec3   delta   = closest - sphC;   // sphere → robot axis
    double dist    = delta.norm();
    double sd      = dist - rRob - rSph;

    Vec3 n;
    if (dist < 1e-9) {
        n = Vec3(0, 0, 1);
    } else {
        n = delta / dist;  // obstacle(sphere) → robot direction
    }

    DistResult r;
    r.sd  = sd;
    r.p_a = sphC + rSph * n;     // point on obstacle sphere surface
    r.p_b = closest - rRob * n;  // point on robot capsule surface
    r.n   = n;                    // obstacle → robot ✓
    return r;
}

/// Robot Capsule vs Capsule Obstacle
inline DistResult distRobotCapsuleVsCapsule(
        const Vec3& robP, const Vec3& robQ, double rRob,
        const Vec3& obsP, const Vec3& obsQ, double rObs) {
    Vec3 c1, c2;
    closestPointsSegSeg(robP, robQ, obsP, obsQ, c1, c2);
    Vec3   delta = c1 - c2;   // obstacle → robot
    double dist  = delta.norm();
    double sd    = dist - rRob - rObs;

    Vec3 n;
    if (dist < 1e-9) {
        n = Vec3(0, 0, 1);
    } else {
        n = delta / dist;
    }

    DistResult r;
    r.sd  = sd;
    r.p_a = c2 + rObs * n;   // point on obstacle capsule surface
    r.p_b = c1 - rRob * n;   // point on robot capsule surface
    r.n   = n;                 // obstacle → robot ✓
    return r;
}

/// Robot Capsule vs Box (OBB) Obstacle
inline DistResult distRobotCapsuleVsBox(
        const Vec3& robP, const Vec3& robQ, double rRob,
        const Vec3& boxCenter, const Mat3& boxR, const Vec3& halfExtents) {
    // Sample points along robot capsule axis and find closest to OBB
    // Simple approach: find closest point on segment to OBB
    // For each sample t, project to OBB, pick minimum
    // More rigorous: iterate from segment endpoints + GJK-like refinement
    // Here we use a robust iterative approach.

    const int N = 8;
    double best_sd = 1e9;
    Vec3   best_p_a = Vec3::Zero(), best_p_b = Vec3::Zero(), best_n = Vec3(0,0,1);

    for (int i = 0; i <= N; ++i) {
        double t = static_cast<double>(i) / N;
        Vec3 q = robP + t * (robQ - robP);
        Vec3 closest_box = closestPointOnOBB(boxCenter, boxR, halfExtents, q);
        Vec3 delta = q - closest_box;  // obstacle → robot
        double dist = delta.norm();
        double sd = dist - rRob;

        if (sd < best_sd) {
            best_sd = sd;
            Vec3 n = (dist < 1e-9) ? Vec3(0, 0, 1) : delta / dist;
            best_p_a = closest_box;          // point on box surface
            best_p_b = q - rRob * n;         // point on robot capsule surface
            best_n   = n;                     // obstacle → robot ✓
        }
    }

    DistResult r;
    r.sd  = best_sd;
    r.p_a = best_p_a;
    r.p_b = best_p_b;
    r.n   = best_n;
    return r;
}
