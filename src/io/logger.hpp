#pragma once
#include <string>

struct PlanResult;   // forward decl
struct Scene;

void writeLog(const std::string& path,
              const Scene& scene,
              const PlanResult& result,
              bool trajopt_ran);
