#include "trajopt.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

using Q7   = Eigen::Matrix<double, 7, 1>;
using Traj = std::vector<Q7>;

TrajOpt::TrajOpt(const Scene& scene, const CollisionChecker& checker)
    : scene_(scene), checker_(checker) {}

// ─────────────────────────────────────────────────────────────────────────────
Q7 TrajOpt::clamp(const Q7& q) const {
    Q7 out = q;
    for (int i = 0; i < 7; ++i)
        out[i] = std::clamp(q[i], scene_.joint_limits[i].min,
                                   scene_.joint_limits[i].max);
    return out;
}

Traj TrajOpt::resample(const std::vector<Q7>& path, int n) const {
    if (path.empty()) return {};
    if (static_cast<int>(path.size()) == n) return path;

    Traj out;
    out.reserve(n);
    int N = static_cast<int>(path.size()) - 1;
    for (int i = 0; i < n; ++i) {
        double t    = static_cast<double>(i) / (n - 1) * N;
        int    idx  = std::min(static_cast<int>(t), N - 1);
        double frac = t - idx;
        out.push_back(path[idx] + frac * (path[std::min(idx+1, N)] - path[idx]));
    }
    return out;
}

double TrajOpt::smoothCost(const Traj& traj) const {
    double cost = 0.0;
    int n = static_cast<int>(traj.size());
    for (int k = 1; k < n - 1; ++k) {
        Q7 acc = traj[k+1] - 2*traj[k] + traj[k-1];
        cost += acc.squaredNorm();
    }
    return cost;
}

double TrajOpt::collisionCost(const Traj& traj) const {
    double cost = 0.0;
    for (const auto& q : traj) {
        DistanceQuery dq;
        dq.q      = q;
        dq.d_safe = scene_.d_safe;
        double sd = checker_.query(dq).min_sd;
        double h  = scene_.d_safe - sd;  // hinge: positive when too close
        if (h > 0) cost += h;
    }
    return cost;
}

double TrajOpt::totalCost(const Traj& traj, double mu) const {
    return scene_.trajopt.w_smooth    * smoothCost(traj)
         + mu * scene_.trajopt.w_collision * collisionCost(traj);
}

Traj TrajOpt::gradient(const Traj& traj, double mu) const {
    const double eps = 1e-4;
    int n = static_cast<int>(traj.size());
    Traj grad(n, Q7::Zero());

    for (int k = 1; k < n - 1; ++k) {  // fix endpoints
        for (int j = 0; j < 7; ++j) {
            Traj tp = traj, tm = traj;
            tp[k][j] += eps;
            tm[k][j] -= eps;
            grad[k][j] = (totalCost(tp, mu) - totalCost(tm, mu)) / (2*eps);
        }
    }
    return grad;
}

// ─────────────────────────────────────────────────────────────────────────────
TrajOptResult TrajOpt::optimize(const std::vector<Q7>& initial_path) {
    const auto& cfg = scene_.trajopt;

    Traj traj = resample(initial_path, cfg.num_waypoints);

    double mu = cfg.mu0;
    TrajOptResult res;

    for (int outer = 0; outer < cfg.outer_iters; ++outer) {
        // Inner: gradient descent with trust-region step size
        double step = cfg.tr_radius0;
        double prev_cost = totalCost(traj, mu);

        for (int inner = 0; inner < 50; ++inner) {
            Traj grad = gradient(traj, mu);

            // Compute gradient norm
            double gnorm = 0;
            for (auto& g : grad) gnorm += g.squaredNorm();
            gnorm = std::sqrt(gnorm);
            if (gnorm < 1e-8) break;

            // Tentative step
            Traj traj_new = traj;
            for (int k = 1; k < static_cast<int>(traj.size()) - 1; ++k)
                traj_new[k] = clamp(traj[k] - step * grad[k]);

            double new_cost = totalCost(traj_new, mu);
            double ratio = (prev_cost - new_cost) / (step * gnorm * gnorm + 1e-12);

            if (ratio > 0.0) {
                traj = traj_new;
                prev_cost = new_cost;
                if (ratio > 0.75) step = std::min(step * cfg.tr_grow,  1.0);
            } else {
                step *= cfg.tr_shrink;
                if (step < 1e-8) break;
            }
        }

        mu *= cfg.mu_mult;
        res.outer_iters_done = outer + 1;
    }

    // Compute final min_sd
    double min_sd = 1e9;
    for (const auto& q : traj) {
        DistanceQuery dq; dq.q = q; dq.d_safe = scene_.d_safe;
        double sd = checker_.query(dq).min_sd;
        if (sd < min_sd) min_sd = sd;
    }

    res.waypoints = traj;
    res.min_sd    = min_sd;
    res.cost      = totalCost(traj, mu / cfg.mu_mult);
    res.success   = (min_sd >= 0.0);
    return res;
}
