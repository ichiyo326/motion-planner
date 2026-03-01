#include <iostream>
#include <fstream>
#include <filesystem>
#include "io/scene_loader.hpp"
#include "io/logger.hpp"
#include "collision/collision_checker.hpp"
#include "planner/rrtstar.hpp"
#include "trajopt/trajopt.hpp"

namespace fs = std::filesystem;

static void saveCsv(const std::string& path,
                    const std::vector<Eigen::Matrix<double,7,1>>& wps) {
    std::ofstream f(path);
    f << "q0,q1,q2,q3,q4,q5,q6\n";
    for (const auto& q : wps) {
        for (int i = 0; i < 7; ++i) {
            f << q[i];
            if (i < 6) f << ",";
        }
        f << "\n";
    }
}

int main(int argc, char* argv[]) {
    std::string scene_path = (argc > 1) ? argv[1] : "scene.json";

    // ── Load ─────────────────────────────────────────────────────────────
    Scene scene;
    try {
        scene = loadScene(scene_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading scene: " << e.what() << "\n";
        return 1;
    }
    std::cout << "[scene]  hash=" << scene.scene_hash << "\n";
    std::cout << "[scene]  obstacles=" << scene.obstacles.size()
              << "  link_capsules=" << scene.link_capsules.size() << "\n";

    // ── Plan ─────────────────────────────────────────────────────────────
    CollisionChecker checker(scene);
    RRTStar planner(scene, checker);

    std::cout << "[rrt*]   seed=" << scene.rrtstar.seed
              << "  max_iter=" << scene.rrtstar.max_iter << "\n";

    PlanResult result = planner.plan(scene.q_start, scene.q_goal);

    if (!result.success) {
        std::cerr << "[rrt*]   FAILED after " << result.nodes << " nodes ("
                  << result.time_sec << "s)\n";
        writeLog("out/plan.log", scene, result, false);
        return 2;
    }

    std::cout << "[rrt*]   SUCCESS  cost=" << result.cost
              << "  nodes=" << result.nodes
              << "  time=" << result.time_sec << "s"
              << "  min_sd=" << result.min_sd << "m\n";

    // ── TrajOpt ──────────────────────────────────────────────────────────
    bool trajopt_ran = false;
    if (scene.trajopt.enabled) {
        std::cout << "[trajopt] running ("
                  << scene.trajopt.outer_iters << " outer iters)...\n";
        TrajOpt to(scene, checker);
        TrajOptResult tor = to.optimize(result.path);

        std::cout << "[trajopt] cost=" << tor.cost
                  << "  min_sd=" << tor.min_sd
                  << "  iters=" << tor.outer_iters_done << "\n";
        if (tor.success) {
            result.path   = tor.waypoints;
            result.min_sd = tor.min_sd;
        }
        trajopt_ran = true;
    }

    // ── Output ───────────────────────────────────────────────────────────
    fs::create_directories("out");
    saveCsv("out/trajectory.csv", result.path);
    writeLog("out/plan.log", scene, result, trajopt_ran);

    std::cout << "[output] out/trajectory.csv  (" << result.path.size()
              << " waypoints)\n";
    std::cout << "[output] out/plan.log\n";
    std::cout << "[render] run:  python tools/render.py --csv out/trajectory.csv "
              << "--scene " << scene_path << " --output out/plan.gif\n";

    return 0;
}
