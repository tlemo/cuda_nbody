
#pragma once

#include <common/math_2d.h>

#include <vector>
#include <string>

constexpr Scalar kMaxCoord = 1e3;
constexpr Scalar kMinBodyMass = 0.1;
constexpr Scalar kMaxBodyMass = 10.0;
constexpr Scalar kTimeStep = 0.1;
constexpr Scalar kDampingFactor = 0.9995;
constexpr Scalar kSofteningFactor = 0.1;

struct Body {
  Vector2 pos;
  Vector2 v;
  Scalar mass = 0;
};

class NBodyPlugin {
 public:
  explicit NBodyPlugin(const std::string& name);

  virtual ~NBodyPlugin() = default;

  NBodyPlugin(const NBodyPlugin&) = delete;
  NBodyPlugin& operator=(const NBodyPlugin&) = delete;

  virtual void Init(const std::vector<Body>& bodies, int width, int height) = 0;
  virtual void Render() = 0;
  virtual void Update() = 0;
  virtual void Shutdown() = 0;
};

void RegisterPlugin(NBodyPlugin* plugin, const std::string& name);

NBodyPlugin* SelectPlugin(const std::string& name);

std::vector<std::string> AvailablePlugins();