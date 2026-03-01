#pragma once
#include <Eigen/Core>
#include "contact.hpp"
#include "io/scene_loader.hpp"

struct DistanceQuery {
    Eigen::Matrix<double, 7, 1> q;
    double d_safe = 0.02;
};

class CollisionChecker {
public:
    explicit CollisionChecker(const Scene& scene);

    /// Query signed distance for a single configuration.
    CollisionResult query(const DistanceQuery& Q) const;

    /// Adaptive-bisection edge collision check.
    /// Returns true if the entire segment q0→q1 is collision-free.
    /// If worst != nullptr, writes the worst contact found along the edge.
    bool checkMotion(
        const Eigen::Matrix<double, 7, 1>& q0,
        const Eigen::Matrix<double, 7, 1>& q1,
        const EdgeCheckConfig& cfg,
        CollisionResult* worst = nullptr
    ) const;

private:
    const Scene& scene_;

    bool checkMotionImpl(
        const Eigen::Matrix<double, 7, 1>& q0,
        const Eigen::Matrix<double, 7, 1>& q1,
        const EdgeCheckConfig& cfg,
        int depth,
        CollisionResult* worst
    ) const;
};
