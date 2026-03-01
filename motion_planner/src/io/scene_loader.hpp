#pragma once
#include <string>
#include <vector>
#include <array>
#include <Eigen/Core>
#include "geometry/primitives.hpp"

// ─────────────────────────────────────────────────────────────────────────────
//  Scene — loaded once from scene.json.
//  This struct is the single source of truth for all modules.
// ─────────────────────────────────────────────────────────────────────────────

struct JointLimit {
    double min{}, max{};
};

struct EdgeCheckConfig {
    double max_step  = 0.05;   ///< max joint-space step [rad]
    int    max_depth = 12;
};

struct RRTStarConfig {
    int    max_iter  = 20000;
    double eta       = 0.3;
    double goal_bias = 0.05;
    double gamma     = 2.0;
    bool   informed  = false;
    int    seed      = 42;
};

struct TrajOptConfig {
    bool   enabled       = false;
    int    num_waypoints = 40;
    double dt            = 0.1;
    double w_smooth      = 1.0;
    double w_collision   = 10.0;
    double mu0           = 1.0;
    double mu_mult       = 10.0;
    int    outer_iters   = 5;
    double tr_radius0    = 0.2;
    double tr_shrink     = 0.5;
    double tr_grow       = 1.5;
};

struct Scene {
    // Robot
    std::vector<JointLimit>  joint_limits;   ///< size 7
    std::vector<LinkCapsule> link_capsules;

    // Start / Goal
    Eigen::Matrix<double, 7, 1> q_start;
    Eigen::Matrix<double, 7, 1> q_goal;
    double goal_tolerance_l2 = 0.05;

    // Obstacles
    std::vector<Obstacle> obstacles;

    // Planning params
    double          d_safe = 0.02;
    EdgeCheckConfig edge_check;
    RRTStarConfig   rrtstar;
    TrajOptConfig   trajopt;

    // Meta
    std::string scene_hash;  ///< SHA-256 of the raw JSON bytes
};

/// Load and validate scene.json.  Throws std::runtime_error on any violation.
Scene loadScene(const std::string& path);
