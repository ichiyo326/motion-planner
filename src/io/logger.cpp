#include "logger.hpp"
#include "scene_loader.hpp"
#include "planner/rrtstar.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void writeLog(const std::string& path,
              const Scene& scene,
              const PlanResult& result,
              bool trajopt_ran) {
    json log;
    log["scene_hash"] = scene.scene_hash;
    log["seed"]       = scene.rrtstar.seed;

    log["planner"] = {
        {"type",       "rrtstar"},
        {"gamma",      scene.rrtstar.gamma},
        {"eta",        scene.rrtstar.eta},
        {"goal_bias",  scene.rrtstar.goal_bias},
        {"max_iter",   scene.rrtstar.max_iter},
        {"informed",   scene.rrtstar.informed},
        {"edge_check", {
            {"mode",      "adaptive"},
            {"max_step",  scene.edge_check.max_step},
            {"max_depth", scene.edge_check.max_depth}
        }}
    };

    log["result"] = {
        {"success",   result.success},
        {"time_sec",  result.time_sec},
        {"nodes",     result.nodes},
        {"best_cost", result.cost},
        {"min_sd",    result.min_sd},
        {"min_sd_note", "signed distance: >0=safe, <0=penetration"},
        {"trajopt_applied", trajopt_ran}
    };

    std::ofstream ofs(path);
    ofs << log.dump(2);
}
