#pragma once
#include <vector>
#include <Eigen/Core>
#include <random>
#include "collision/collision_checker.hpp"
#include "io/scene_loader.hpp"

struct PlanResult {
    bool   success   = false;
    double cost      = 1e9;
    std::vector<Eigen::Matrix<double, 7, 1>> path;  ///< joint-space waypoints
    int    nodes     = 0;
    double time_sec  = 0.0;
    double min_sd    = 1e9;   ///< minimum signed distance along path
};

class RRTStar {
public:
    RRTStar(const Scene& scene, const CollisionChecker& checker);

    PlanResult plan(
        const Eigen::Matrix<double, 7, 1>& q_start,
        const Eigen::Matrix<double, 7, 1>& q_goal
    );

private:
    const Scene&            scene_;
    const CollisionChecker& checker_;

    using Q7 = Eigen::Matrix<double, 7, 1>;

    std::vector<Q7>    nodes_;
    std::vector<int>   parents_;
    std::vector<double>costs_;

    int  nearest(const Q7& q) const;
    std::vector<int> near(const Q7& q, double r) const;
    Q7   steer(const Q7& from, const Q7& to) const;
    Q7   sampleUniform(std::mt19937& rng) const;
    Q7   sampleInformed(std::mt19937& rng, const Q7& q_start,
                        const Q7& q_goal, double c_best) const;
};
