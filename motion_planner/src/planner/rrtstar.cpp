#include "rrtstar.hpp"
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <limits>

using Q7 = Eigen::Matrix<double, 7, 1>;

RRTStar::RRTStar(const Scene& scene, const CollisionChecker& checker)
    : scene_(scene), checker_(checker) {}

// ─────────────────────────────────────────────────────────────────────────────
int RRTStar::nearest(const Q7& q) const {
    int   best = 0;
    double best_d = (nodes_[0] - q).norm();
    for (int i = 1; i < static_cast<int>(nodes_.size()); ++i) {
        double d = (nodes_[i] - q).norm();
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

std::vector<int> RRTStar::near(const Q7& q, double r) const {
    std::vector<int> result;
    for (int i = 0; i < static_cast<int>(nodes_.size()); ++i) {
        if ((nodes_[i] - q).norm() <= r)
            result.push_back(i);
    }
    return result;
}

Q7 RRTStar::steer(const Q7& from, const Q7& to) const {
    double eta = scene_.rrtstar.eta;
    Q7 diff = to - from;
    double d = diff.norm();
    if (d <= eta) return to;
    return from + (eta / d) * diff;
}

Q7 RRTStar::sampleUniform(std::mt19937& rng) const {
    Q7 q;
    for (int i = 0; i < 7; ++i) {
        std::uniform_real_distribution<double> dist(
            scene_.joint_limits[i].min, scene_.joint_limits[i].max);
        q[i] = dist(rng);
    }
    return q;
}

Q7 RRTStar::sampleInformed(std::mt19937& rng, const Q7& q_start,
                             const Q7& q_goal, double c_best) const {
    // Prolate hyperspheroid sampling in joint space
    double c_min = (q_goal - q_start).norm();
    if (c_best <= c_min + 1e-9) return sampleUniform(rng);

    Q7 centre = 0.5 * (q_start + q_goal);
    double a = c_best / 2.0;
    double b = std::sqrt(std::max(0.0, c_best*c_best - c_min*c_min)) / 2.0;

    // Sample from ball, scale to ellipsoid
    std::normal_distribution<double> nd(0.0, 1.0);
    Q7 ball;
    for (int i = 0; i < 7; ++i) ball[i] = nd(rng);
    ball.normalize();

    // Scale
    Q7 r_ball = ball;
    r_ball[0] *= a;
    for (int i = 1; i < 7; ++i) r_ball[i] *= b;

    Q7 q = centre + r_ball;

    // Clamp to joint limits
    for (int i = 0; i < 7; ++i)
        q[i] = std::clamp(q[i], scene_.joint_limits[i].min, scene_.joint_limits[i].max);
    return q;
}

// ─────────────────────────────────────────────────────────────────────────────
PlanResult RRTStar::plan(const Q7& q_start, const Q7& q_goal) {
    auto t_start = std::chrono::steady_clock::now();

    std::mt19937 rng(static_cast<uint32_t>(scene_.rrtstar.seed));
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    nodes_.clear(); parents_.clear(); costs_.clear();
    nodes_.push_back(q_start);
    parents_.push_back(0);
    costs_.push_back(0.0);

    const double gamma     = scene_.rrtstar.gamma;
    const double eta       = scene_.rrtstar.eta;
    const double goal_bias = scene_.rrtstar.goal_bias;
    const double tol       = scene_.goal_tolerance_l2;
    const bool   informed  = scene_.rrtstar.informed;
    const int    max_iter  = scene_.rrtstar.max_iter;
    const EdgeCheckConfig& ec = scene_.edge_check;

    int    best_goal_idx = -1;
    double best_cost     = std::numeric_limits<double>::infinity();

    for (int iter = 0; iter < max_iter; ++iter) {
        int n = static_cast<int>(nodes_.size());

        // Sample
        Q7 q_rand;
        if (uni01(rng) < goal_bias) {
            q_rand = q_goal;
        } else if (informed && best_goal_idx >= 0) {
            q_rand = sampleInformed(rng, q_start, q_goal, best_cost);
        } else {
            q_rand = sampleUniform(rng);
        }

        // Nearest + steer
        int q_near_idx = nearest(q_rand);
        Q7 q_new = steer(nodes_[q_near_idx], q_rand);

        // Collision check
        if (!checker_.checkMotion(nodes_[q_near_idx], q_new, ec))
            continue;

        // Near radius: r(n) = min(gamma*(log(n)/n)^(1/7), eta)
        double r = (n > 1)
            ? std::min(gamma * std::pow(std::log((double)n) / n, 1.0/7.0), eta)
            : eta;
        auto   near_ids = near(q_new, r);

        // Choose best parent
        double cost_new = costs_[q_near_idx] + (q_new - nodes_[q_near_idx]).norm();
        int    parent   = q_near_idx;

        for (int nid : near_ids) {
            double candidate = costs_[nid] + (q_new - nodes_[nid]).norm();
            if (candidate < cost_new &&
                checker_.checkMotion(nodes_[nid], q_new, ec)) {
                cost_new = candidate;
                parent   = nid;
            }
        }

        nodes_.push_back(q_new);
        parents_.push_back(parent);
        costs_.push_back(cost_new);
        int new_idx = static_cast<int>(nodes_.size()) - 1;

        // Rewire
        for (int nid : near_ids) {
            double new_cost = cost_new + (nodes_[nid] - q_new).norm();
            if (new_cost < costs_[nid] &&
                checker_.checkMotion(q_new, nodes_[nid], ec)) {
                parents_[nid] = new_idx;
                costs_[nid]   = new_cost;
            }
        }

        // Goal check
        if ((q_new - q_goal).norm() < tol && cost_new < best_cost) {
            best_cost     = cost_new;
            best_goal_idx = new_idx;
        }
    }

    // ── Extract result ────────────────────────────────────────────────────
    PlanResult res;
    res.nodes    = static_cast<int>(nodes_.size());

    auto t_end = std::chrono::steady_clock::now();
    res.time_sec = std::chrono::duration<double>(t_end - t_start).count();

    if (best_goal_idx < 0) {
        res.success = false;
        return res;
    }

    // Trace path
    std::vector<Q7> path;
    int idx = best_goal_idx;
    while (idx != 0) {
        path.push_back(nodes_[idx]);
        idx = parents_[idx];
    }
    path.push_back(q_start);
    std::reverse(path.begin(), path.end());

    // Compute min_sd along path
    double min_sd = 1e9;
    for (const auto& q : path) {
        DistanceQuery dq;
        dq.q      = q;
        dq.d_safe = scene_.d_safe;
        double sd = checker_.query(dq).min_sd;
        if (sd < min_sd) min_sd = sd;
    }

    res.success = true;
    res.cost    = best_cost;
    res.path    = std::move(path);
    res.min_sd  = min_sd;
    return res;
}
