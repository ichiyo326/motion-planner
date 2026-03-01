#include "collision_checker.hpp"
#include "geometry/distance.hpp"
#include "kinematics/fk.hpp"
#include <variant>

CollisionChecker::CollisionChecker(const Scene& scene) : scene_(scene) {}

// ─────────────────────────────────────────────────────────────────────────────
CollisionResult CollisionChecker::query(const DistanceQuery& Q) const {
    FKResult fk = computeFK(Q.q);

    CollisionResult result;
    result.min_sd = 1e9;

    const auto& caps = scene_.link_capsules;
    const auto& obs  = scene_.obstacles;

    for (int li = 0; li < static_cast<int>(caps.size()); ++li) {
        // Capsule endpoints in world frame
        const Vec3 P = fk[caps[li].joint_i].pos;
        const Vec3 Q_pt = fk[caps[li].joint_j].pos;
        double rRob = caps[li].radius;

        for (int oi = 0; oi < static_cast<int>(obs.size()); ++oi) {
            DistResult dr;

            std::visit([&](const auto& o) {
                using T = std::decay_t<decltype(o)>;
                if constexpr (std::is_same_v<T, SphereObs>) {
                    dr = distRobotCapsuleVsSphere(P, Q_pt, rRob,
                                                  o.pos, o.radius);
                } else if constexpr (std::is_same_v<T, BoxObs>) {
                    Mat3 R = toMatrix(o.quat);
                    dr = distRobotCapsuleVsBox(P, Q_pt, rRob,
                                               o.pos, R, o.half_extents);
                } else if constexpr (std::is_same_v<T, CapsuleObs>) {
                    // Capsule obstacle endpoints
                    Vec3 axis = o.quat * Vec3(0, 0, 1);
                    Vec3 obsP = o.pos - o.half_length * axis;
                    Vec3 obsQ = o.pos + o.half_length * axis;
                    dr = distRobotCapsuleVsCapsule(P, Q_pt, rRob,
                                                   obsP, obsQ, o.radius);
                }
            }, obs[oi]);

            if (dr.sd < result.min_sd) {
                result.min_sd = dr.sd;
                result.worst.sd      = dr.sd;
                result.worst.n       = dr.n;
                result.worst.p_obs   = dr.p_a;
                result.worst.p_robot = dr.p_b;
                result.worst.link_id = li;
                result.worst.obs_id  = oi;
            }
        }
    }

    result.in_collision = (result.min_sd < 0.0);
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
bool CollisionChecker::checkMotionImpl(
        const Eigen::Matrix<double, 7, 1>& q0,
        const Eigen::Matrix<double, 7, 1>& q1,
        const EdgeCheckConfig& cfg,
        int depth,
        CollisionResult* worst) const {

    Eigen::Matrix<double, 7, 1> qmid = 0.5 * (q0 + q1);
    DistanceQuery dq;
    dq.q      = qmid;
    dq.d_safe = scene_.d_safe;

    CollisionResult r = query(dq);
    if (worst && r.min_sd < worst->min_sd) *worst = r;
    if (r.in_collision) return false;

    if (depth >= cfg.max_depth) return true;

    double dist = (q1 - q0).norm();
    if (dist <= cfg.max_step) return true;

    return checkMotionImpl(q0, qmid, cfg, depth + 1, worst)
        && checkMotionImpl(qmid, q1,  cfg, depth + 1, worst);
}

bool CollisionChecker::checkMotion(
        const Eigen::Matrix<double, 7, 1>& q0,
        const Eigen::Matrix<double, 7, 1>& q1,
        const EdgeCheckConfig& cfg,
        CollisionResult* worst) const {
    // Also check endpoints
    DistanceQuery d0; d0.q = q0; d0.d_safe = scene_.d_safe;
    DistanceQuery d1; d1.q = q1; d1.d_safe = scene_.d_safe;
    auto r0 = query(d0);
    auto r1 = query(d1);
    if (worst) {
        if (r0.min_sd < worst->min_sd) *worst = r0;
        if (r1.min_sd < worst->min_sd) *worst = r1;
    }
    if (r0.in_collision || r1.in_collision) return false;

    return checkMotionImpl(q0, q1, cfg, 0, worst);
}
