
#include <common/nbody_plugin.h>
#include <common/utils.h>
#include <common/math_2d.h>

#include <GL/gl.h>

#include <utility>
#include <vector>
#include <math.h>

class DummyNBody : public NBodyPlugin {
 public:
  DummyNBody() : NBodyPlugin("dummy") {}

 private:
  void Init(const std::vector<Body>& bodies, int, int) final { bodies_ = bodies; }

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

  void Update() final {}

 private:
  std::vector<Body> bodies_;
};

static DummyNBody instance;
