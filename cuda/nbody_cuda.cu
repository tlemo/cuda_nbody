
#include <common/cuda_utils.h>
#include <common/math_2d.h>
#include <common/nbody_plugin.h>
#include <common/utils.h>

#include <GL/gl.h>

#include <utility>
#include <vector>
#include <math.h>

namespace cuda_naive {

__global__ static void UpdateKernel(Body* bodies, const Body* prev_bodies, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    Body body = prev_bodies[index];
    Vector2 acc;
    for (int j = 0; j < count; ++j) {
      const auto& other = prev_bodies[j];
      const Vector2 r = other.pos - body.pos;
      const Scalar dist_squared = length_squared(r) + kSofteningFactor;
      const Scalar inv_dist = 1 / sqrt(dist_squared);
      const Scalar inv_dist_cube = inv_dist * inv_dist * inv_dist;
      const Scalar s = other.mass * inv_dist_cube;
      acc = acc + r * s;
    }
    body.v = body.v + acc * kTimeStep;
    body.v = body.v * kDampingFactor;
    body.pos = body.pos + body.v * kTimeStep;
    bodies[index] = body;
  }
}

class NBody : public NBodyPlugin {
 public:
  NBody() : NBodyPlugin("cuda") {}

 private:
  void Free() {
    CU(cudaFree(bodies_));
    CU(cudaFree(prev_bodies_));
    bodies_ = nullptr;
    prev_bodies_ = nullptr;
    bodies_count_ = 0;
  }

  void Init(const std::vector<Body>& bodies, int, int) final {
    PrintCudaInfo();
    Free();

    bodies_count_ = bodies.size();
    CHECK(bodies_count_ > 0);
    const size_t buffer_size = bodies_count_ * sizeof(Body);

    CU(cudaMallocManaged(&bodies_, buffer_size));
    CU(cudaMallocManaged(&prev_bodies_, buffer_size));
    CU(cudaMemcpy(bodies_, bodies.data(), buffer_size, cudaMemcpyHostToDevice));
  }

  void Shutdown() final { Free(); }

  void Render() final {
    glColor3f(0.5, 1.0, 0.9);
    glPointSize(1.0);
    glBegin(GL_POINTS);
    for (int i = 0; i < bodies_count_; ++i) {
      glVertex2f(bodies_[i].pos.x, bodies_[i].pos.y);
    }
    glEnd();
  }

  void Update() final {
    std::swap(bodies_, prev_bodies_);

    const int kBlockSize = 128;
    const int kBlockCount = (bodies_count_ + kBlockSize - 1) / kBlockSize;
    UpdateKernel<<<kBlockCount, kBlockSize>>>(bodies_, prev_bodies_, bodies_count_);
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
  }

 private:
  Body* prev_bodies_ = nullptr;
  Body* bodies_ = nullptr;
  int bodies_count_ = 0;
};

static NBody instance;

}  // namespace cuda_naive
