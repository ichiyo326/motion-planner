#pragma once
#include <string>
#include <variant>
#include "math/vec3.hpp"
#include "math/quat.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  Shape size convention (see README / Design.md §2.4):
//    sphere  : radius (m)
//    box     : half_extents [hx, hy, hz]  ← HALF-side (full = 2×half)
//    capsule : radius + half_length        ← HALF of axis (full = 2×half_length)
// ─────────────────────────────────────────────────────────────────────────────

struct SphereObs {
    std::string id;
    Vec3   pos;
    double radius{};
};

struct BoxObs {
    std::string id;
    Vec3 pos;
    Quat quat;
    Vec3 half_extents;  ///< [hx, hy, hz] — each is HALF the full side length
};

struct CapsuleObs {
    std::string id;
    Vec3   pos;
    Quat   quat;
    double radius{};
    double half_length{};   ///< half of axis length (endpoints excluded)
};

using Obstacle = std::variant<SphereObs, BoxObs, CapsuleObs>;

/// Robot link represented as a capsule defined by two joint-frame origins.
struct LinkCapsule {
    int    joint_i{};
    int    joint_j{};
    double radius{};
};
