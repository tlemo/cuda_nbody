
// rsqrt + structure-of-arrays (SoA)

#include <common/cuda_utils.h>
#include <common/math_2d.h>
#include <common/nbody_plugin.h>
#include <common/utils.h>

#include <GL/gl.h>

#include <utility>
#include <vector>
#include <math.h>

namespace cuda_soa {

struct KernelArgs {
  Vector2* prev_pos = nullptr;
  Vector2* pos = nullptr;
  Vector2* vel = nullptr;
  Scalar* mass = nullptr;
  int count = 0;
};

__global__ static void UpdateKernel(KernelArgs args) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < args.count) {
    const auto pos = args.prev_pos[index];
    Vector2 acc = { 0.0f, 0.0f };
    for (int j = 0; j < args.count; ++j) {
      const Vector2 r = args.prev_pos[j] - pos;
      const Scalar dist_squared = length_squared(r) + kSofteningFactor;
      const Scalar inv_dist = rsqrtf(dist_squared);
      const Scalar inv_dist_cube = inv_dist * inv_dist * inv_dist;
      const Scalar s = args.mass[j] * inv_dist_cube;
      acc = acc + r * s;
    }
    const auto v = (args.vel[index] + acc * kTimeStep) * kDampingFactor;
    args.pos[index] = pos + v * kTimeStep;
    args.vel[index] = v;
  }
}

class NBody : public NBodyPlugin {
 public:
  NBody() : NBodyPlugin("cuda_soa") {}

 private:
  void Free() {
    CU(cudaFree(pos_));
    CU(cudaFree(prev_pos_));
    CU(cudaFree(vel_));
    CU(cudaFree(mass_));
    pos_ = nullptr;
    prev_pos_ = nullptr;
    vel_ = nullptr;
    mass_ = nullptr;
    bodies_count_ = 0;
  }

  void Init(const std::vector<Body>& bodies, int, int) final {
    PrintCudaInfo();

    // reset state
    Free();

    bodies_count_ = bodies.size();
    CHECK(bodies_count_ > 0);
    const size_t buffer_size = bodies_count_ * sizeof(Vector2);

    CU(cudaMallocManaged(&pos_, buffer_size));
    CU(cudaMallocManaged(&prev_pos_, buffer_size));
    CU(cudaMallocManaged(&vel_, buffer_size));
    CU(cudaMallocManaged(&mass_, bodies_count_ * sizeof(Scalar)));

    for (int i = 0; i < bodies_count_; ++i) {
      pos_[i].x = bodies[i].pos.x;
      pos_[i].y = bodies[i].pos.y;
      vel_[i].x = bodies[i].v.x;
      vel_[i].y = bodies[i].v.y;
      mass_[i] = bodies[i].mass;
    }
  }

  void Shutdown() final { Free(); }

  void Render() final {
    glColor3f(0.5, 1.0, 0.9);
    glPointSize(1.0);
    glBegin(GL_POINTS);
    for (int i = 0; i < bodies_count_; ++i) {
      glVertex2f(pos_[i].x, pos_[i].y);
    }
    glEnd();
  }

  void Update() final {
    std::swap(pos_, prev_pos_);

    KernelArgs args;
    args.prev_pos = prev_pos_;
    args.pos = pos_;
    args.vel = vel_;
    args.mass = mass_;
    args.count = bodies_count_;

    const int kBlockSize = 128;
    const int kBlockCount = CeilDiv(bodies_count_, kBlockSize);
    UpdateKernel<<<kBlockCount, kBlockSize>>>(args);

    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
  }

 private:
  Vector2* prev_pos_ = nullptr;
  Vector2* pos_ = nullptr;
  Vector2* vel_ = nullptr;
  Scalar* mass_ = nullptr;
  int bodies_count_ = 0;
};

static NBody instance;

}  // namespace cuda_soa
