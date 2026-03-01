#pragma once
#include <vector>
#include <Eigen/Core>
#include "collision/collision_checker.hpp"
#include "io/scene_loader.hpp"

struct TrajOptResult {
    bool   success         = false;
    double cost            = 1e9;
    double min_sd          = -1e9;
    int    outer_iters_done = 0;
    std::vector<Eigen::Matrix<double, 7, 1>> waypoints;
};

class TrajOpt {
public:
    TrajOpt(const Scene& scene, const CollisionChecker& checker);

    /// Optimise an initial path from RRT*.
    /// Returns refined waypoints with improved smoothness and clearance.
    TrajOptResult optimize(
        const std::vector<Eigen::Matrix<double, 7, 1>>& initial_path
    );

private:
    using Q7  = Eigen::Matrix<double, 7, 1>;
    using Traj = std::vector<Q7>;

    const Scene&            scene_;
    const CollisionChecker& checker_;

    /// Smooth cost: sum of squared finite-differences (acceleration)
    double smoothCost(const Traj& traj) const;

    /// Collision cost with ℓ1 hinge: sum of max(0, d_safe - sd)
    double collisionCost(const Traj& traj) const;

    /// Total cost: w_smooth*smooth + mu*collision
    double totalCost(const Traj& traj, double mu) const;

    /// Gradient of total cost w.r.t. each waypoint (central differences)
    Traj gradient(const Traj& traj, double mu) const;

    /// Project waypoint to joint limits
    Q7 clamp(const Q7& q) const;

    /// Resample path to num_waypoints using linear interpolation
    Traj resample(const std::vector<Q7>& path, int n) const;
};
