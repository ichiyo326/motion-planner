#include "scene_loader.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ── Tiny SHA-256 stub (FNV-64 as placeholder for demo) ───────────────────────
static std::string hashString(const std::string& s) {
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    char buf[20];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
    return std::string("fnv64:") + buf;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
static Vec3 parseVec3(const json& j) {
    return Vec3(j[0].get<double>(), j[1].get<double>(), j[2].get<double>());
}

static Quat parseQuat(const json& j) {
    // JSON order: [w, x, y, z]
    return quatFromWXYZ(j[0].get<double>(), j[1].get<double>(),
                        j[2].get<double>(), j[3].get<double>());
}

// ─────────────────────────────────────────────────────────────────────────────
Scene loadScene(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("Cannot open scene file: " + path);

    std::ostringstream ss;
    ss << ifs.rdbuf();
    std::string raw = ss.str();

    json j = json::parse(raw);
    Scene scene;
    scene.scene_hash = hashString(raw);

    // ── meta validation ──────────────────────────────────────────────────
    if (j.contains("meta")) {
        auto& m = j["meta"];
        if (m.value("quat", "") != "wxyz")
            throw std::runtime_error("scene.json: meta.quat must be 'wxyz'");
    }

    // ── robot ─────────────────────────────────────────────────────────────
    auto& rob = j["robot"];
    auto& jlims = rob["joint_limits"];
    if (jlims.size() != 7)
        throw std::runtime_error("joint_limits must have exactly 7 entries");
    for (auto& lim : jlims)
        scene.joint_limits.push_back({lim["min"].get<double>(),
                                       lim["max"].get<double>()});

    for (auto& lc : rob["link_capsules"]) {
        LinkCapsule cap;
        cap.joint_i = lc["joint_i"].get<int>();
        cap.joint_j = lc["joint_j"].get<int>();
        cap.radius  = lc["radius"].get<double>();
        if (cap.joint_i >= cap.joint_j)
            throw std::runtime_error("link_capsules: joint_i must be < joint_j");
        if (cap.radius <= 0)
            throw std::runtime_error("link_capsules: radius must be > 0");
        scene.link_capsules.push_back(cap);
    }

    // ── start / goal ─────────────────────────────────────────────────────
    {
        auto& qs = j["start"]["q"];
        if (qs.size() != 7) throw std::runtime_error("start.q must have 7 values");
        for (int i = 0; i < 7; ++i) scene.q_start[i] = qs[i].get<double>();

        auto& qg = j["goal"]["q"];
        if (qg.size() != 7) throw std::runtime_error("goal.q must have 7 values");
        for (int i = 0; i < 7; ++i) scene.q_goal[i] = qg[i].get<double>();

        if (j["goal"].contains("tolerance"))
            scene.goal_tolerance_l2 = j["goal"]["tolerance"].value("l2", 0.05);
    }

    // ── obstacles ─────────────────────────────────────────────────────────
    for (auto& obs : j["obstacles"]) {
        std::string type = obs["type"].get<std::string>();
        std::string id   = obs["id"].get<std::string>();

        if (type == "sphere") {
            SphereObs s;
            s.id     = id;
            s.pos    = parseVec3(obs["pos"]);
            s.radius = obs["radius"].get<double>();
            scene.obstacles.push_back(s);
        } else if (type == "box") {
            BoxObs b;
            b.id           = id;
            b.pos          = parseVec3(obs["pos"]);
            b.quat         = parseQuat(obs["quat"]);
            b.half_extents = parseVec3(obs["half_extents"]);
            if ((b.half_extents.array() <= 0).any())
                throw std::runtime_error("box half_extents must all be > 0");
            scene.obstacles.push_back(b);
        } else if (type == "capsule") {
            CapsuleObs c;
            c.id          = id;
            c.pos         = parseVec3(obs["pos"]);
            c.quat        = parseQuat(obs["quat"]);
            c.radius      = obs["radius"].get<double>();
            c.half_length = obs["half_length"].get<double>();
            if (c.half_length <= 0)
                throw std::runtime_error("capsule half_length must be > 0");
            scene.obstacles.push_back(c);
        } else {
            throw std::runtime_error("Unknown obstacle type: " + type);
        }
    }

    // ── planning ──────────────────────────────────────────────────────────
    auto& pl = j["planning"];
    scene.d_safe = pl.value("d_safe", 0.02);
    if (pl.contains("edge_check")) {
        auto& ec = pl["edge_check"];
        scene.edge_check.max_step  = ec.value("max_step",  0.05);
        scene.edge_check.max_depth = ec.value("max_depth", 12);
    }
    if (pl.contains("rrtstar")) {
        auto& r = pl["rrtstar"];
        scene.rrtstar.max_iter  = r.value("max_iter",  20000);
        scene.rrtstar.eta       = r.value("eta",       0.3);
        scene.rrtstar.goal_bias = r.value("goal_bias", 0.05);
        scene.rrtstar.gamma     = r.value("gamma",     2.0);
        scene.rrtstar.informed  = r.value("informed",  false);
        scene.rrtstar.seed      = r.value("seed",      42);
    }

    // ── trajopt ───────────────────────────────────────────────────────────
    if (j.contains("trajopt")) {
        auto& to = j["trajopt"];
        scene.trajopt.enabled      = to.value("enabled",       false);
        scene.trajopt.num_waypoints= to.value("num_waypoints", 40);
        scene.trajopt.dt           = to.value("dt",            0.1);
        if (to.contains("weights")) {
            scene.trajopt.w_smooth    = to["weights"].value("smooth",    1.0);
            scene.trajopt.w_collision = to["weights"].value("collision", 10.0);
        }
        if (to.contains("penalty")) {
            scene.trajopt.mu0         = to["penalty"].value("mu0",         1.0);
            scene.trajopt.mu_mult     = to["penalty"].value("mu_mult",     10.0);
            scene.trajopt.outer_iters = to["penalty"].value("outer_iters", 5);
        }
        if (to.contains("trust_region")) {
            scene.trajopt.tr_radius0 = to["trust_region"].value("radius0", 0.2);
            scene.trajopt.tr_shrink  = to["trust_region"].value("shrink",  0.5);
            scene.trajopt.tr_grow    = to["trust_region"].value("grow",    1.5);
        }
    }

    return scene;
}
