#pragma once
#include "math/vec3.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  Signed Distance Convention (Design.md §4.1):
//    sd > 0  : separated (safe)
//    sd = 0  : contact
//    sd < 0  : penetrating
//
//  normal n : obstacle → robot  (push-out direction)
//  invariant: sd ≈ (p_robot - p_obs).dot(n)
// ─────────────────────────────────────────────────────────────────────────────

struct Contact {
    double sd       = 1e9;                    ///< signed distance [m]
    Vec3   n        = Vec3(0, 0, 1);          ///< unit normal: obstacle → robot
    Vec3   p_obs    = Vec3::Zero();           ///< closest point on obstacle surface
    Vec3   p_robot  = Vec3::Zero();           ///< closest point on robot capsule surface
    int    link_id  = -1;                     ///< robot capsule index
    int    obs_id   = -1;                     ///< obstacle index in scene.obstacles
};

struct CollisionResult {
    bool    in_collision = false;   ///< true when min_sd < 0
    double  min_sd       = 1e9;    ///< minimum signed distance across all pairs
    Contact worst;                  ///< contact with minimum sd
};
