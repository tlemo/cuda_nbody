
#include <common/nbody_plugin.h>
#include <common/utils.h>
#include <common/math_2d.h>

#include <GL/gl.h>

#include <utility>
#include <vector>
#include <math.h>

class OpenMpNBody : public NBodyPlugin {
 public:
  OpenMpNBody() : NBodyPlugin("openmp") {}

 private:
  void Init(const std::vector<Body>& bodies, int, int) final {
    bodies_ = bodies;
    prev_bodies_.resize(bodies_.size());
  }

  void Shutdown() final {}

  void Render() final {
    glColor3f(0.5, 1.0, 0.9);
    glPointSize(1.0);
    glBegin(GL_POINTS);
    for (const auto& body : bodies_) {
      glVertex2f(body.pos.x, body.pos.y);
    }
    glEnd();
  }

  void Update() final {
    std::swap(bodies_, prev_bodies_);

    #pragma omp parallel for
    for (int i = 0; i < bodies_.size(); ++i) {
      Body body = prev_bodies_[i];
      Vector2 acc = { 0, 0 };
      for (const auto& other : prev_bodies_) {
        const Vector2 r = other.pos - body.pos;
        const Scalar dist_squared = length_squared(r) + kSofteningFactor;
        const Scalar inv_dist = 1.0f / sqrtf(dist_squared);
        const Scalar inv_dist_cube = inv_dist * inv_dist * inv_dist;
        const Scalar s = other.mass * inv_dist_cube;
        acc = acc + r * s;
      }
      body.v = body.v + acc * kTimeStep;
      body.v = body.v * kDampingFactor;
      body.pos = body.pos + body.v * kTimeStep;
      bodies_[i] = body;
    }
  }

 private:
  std::vector<Body> prev_bodies_;
  std::vector<Body> bodies_;
};

static OpenMpNBody instance;
