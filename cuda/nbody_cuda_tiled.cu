
// Structure-of-Arrays + VBO + Tiled

#include <common/cuda_utils.h>
#include <common/math_2d.h>
#include <common/nbody_plugin.h>
#include <common/utils.h>

#include <GL/glew.h>
#include <GL/gl.h>

#include <cuda_gl_interop.h>

#include <utility>
#include <vector>
#include <math.h>

namespace cuda_tiled {

constexpr int kTileSize = 128;

struct KernelArgs {
  Vector2* prev_pos = nullptr;
  Vector2* pos = nullptr;
  Vector2* vel = nullptr;
  Scalar* mass = nullptr;
  int count = 0;
};

__shared__ Vector2 pos_cache[kTileSize];

__global__ static void UpdateKernel(KernelArgs args) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const auto pos = args.prev_pos[index];

  Vector2 acc = { 0.0f, 0.0f };

  for (int j = 0; j < args.count; j += kTileSize) {
    // cache prev_pos[j .. j + kTileSize)
    pos_cache[threadIdx.x] = args.prev_pos[j + threadIdx.x];

    __syncthreads();

    // calculate the forces for the current tile
    for (int k = 0; k < kTileSize; ++k) {
      const Vector2 r = pos_cache[k] - pos;
      const Scalar dist_squared = length_squared(r) + kSofteningFactor;
      const Scalar inv_dist = rsqrtf(dist_squared);
      const Scalar inv_dist_cube = inv_dist * inv_dist * inv_dist;
      const Scalar s = args.mass[j] * inv_dist_cube;
      acc = acc + r * s;
    }

    __syncthreads();
  }

  const auto v = (args.vel[index] + acc * kTimeStep) * kDampingFactor;
  args.pos[index] = pos + v * kTimeStep;
  args.vel[index] = v;
}

class NBody : public NBodyPlugin {
 public:
  NBody() : NBodyPlugin("cuda_tiled") {}

 private:
  void Free() {
    if (prev_pos_cu_ != nullptr) {
      CHECK(pos_cu_ != nullptr);
      CU(cudaGraphicsUnregisterResource(prev_pos_cu_));
      CU(cudaGraphicsUnregisterResource(pos_cu_));
      prev_pos_cu_ = nullptr;
      pos_cu_ = nullptr;
    }

    if (prev_pos_vbo_ != 0) {
      CHECK(pos_vbo_ != 0);
      glDeleteBuffers(1, &prev_pos_vbo_);
      glDeleteBuffers(1, &pos_vbo_);
      prev_pos_vbo_ = 0;
      pos_vbo_ = 0;
    }

    CU(cudaFree(vel_));
    CU(cudaFree(mass_));
    vel_ = nullptr;
    mass_ = nullptr;

    bodies_count_ = 0;
  }

  void Init(const std::vector<Body>& bodies, int, int) final {
    PrintCudaInfo();

    // reset state
    Free();

    bodies_count_ = ((bodies.size() + kTileSize - 1) / kTileSize) * kTileSize;
    CHECK(bodies_count_ > 0);

    printf("Tile size: %d\n", kTileSize);
    printf("Rounding up the number of bodies to tile size: %d\n",
           bodies_count_);

    const size_t buffer_size = bodies_count_ * sizeof(Vector2);

    // CUDA buffers
    CU(cudaMallocManaged(&vel_, buffer_size));
    CU(cudaMallocManaged(&mass_, bodies_count_ * sizeof(Scalar)));

    // OpenGL buffers
    GLint actual_size = 0;

    glGenBuffers(1, &prev_pos_vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, prev_pos_vbo_);
    glBufferData(GL_ARRAY_BUFFER, buffer_size, nullptr, GL_DYNAMIC_DRAW);

    // sanity check
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &actual_size);
    CHECK(actual_size == buffer_size);

    glGenBuffers(1, &pos_vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo_);
    glBufferData(GL_ARRAY_BUFFER, buffer_size, nullptr, GL_DYNAMIC_DRAW);

    // sanity check
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &actual_size);
    CHECK(actual_size == buffer_size);

    // finally, setup GL/CUDA interop
    // (this doesn't seem to work after glMapBuffer() was used?)
    // glBindBuffer(GL_ARRAY_BUFFER, 0);
    CU(cudaGraphicsGLRegisterBuffer(
        &prev_pos_cu_, prev_pos_vbo_, cudaGraphicsMapFlagsNone));
    CU(cudaGraphicsGLRegisterBuffer(
        &pos_cu_, pos_vbo_, cudaGraphicsMapFlagsNone));

    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo_);
    auto pos =
        static_cast<Vector2*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
    CHECK(pos != nullptr);

    // copy initial values
    for (int i = 0; i < bodies_count_; ++i) {
      if (i < bodies.size()) {
        pos[i] = bodies[i].pos;
        vel_[i] = bodies[i].v;
        mass_[i] = bodies[i].mass;
      } else {
        pos[i] = { 0, 0 };
        vel_[i] = { 0, 0 };
        mass_[i] = kMinBodyMass;
      }
    }

    CHECK(glUnmapBuffer(GL_ARRAY_BUFFER));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  void Shutdown() final { Free(); }

  void Render() final {
    glColor3f(0.5, 1.0, 0.9);
    glPointSize(1.0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, pos_vbo_);

    glVertexPointer(2, GL_FLOAT, 0, 0);

    glDrawArrays(GL_POINTS, 0, bodies_count_);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
  }

  void Update() final {
    std::swap(pos_vbo_, prev_pos_vbo_);
    std::swap(pos_cu_, prev_pos_cu_);

    // mapping hints
    CU(cudaGraphicsResourceSetMapFlags(pos_cu_,
                                       cudaGraphicsMapFlagsWriteDiscard));
    CU(cudaGraphicsResourceSetMapFlags(prev_pos_cu_,
                                       cudaGraphicsMapFlagsReadOnly));

    // map the resources
    CU(cudaGraphicsMapResources(1, &pos_cu_, 0));
    CU(cudaGraphicsMapResources(1, &prev_pos_cu_, 0));

    KernelArgs args;
    args.vel = vel_;
    args.mass = mass_;
    args.count = bodies_count_;

    size_t unused = 0;
    CU(cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&args.pos), &unused, pos_cu_));
    CU(cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&args.prev_pos), &unused, prev_pos_cu_));

    const int kBlockSize = kTileSize;
    const int kBlockCount = (bodies_count_ + kBlockSize - 1) / kBlockSize;
    UpdateKernel<<<kBlockCount, kBlockSize>>>(args);
    CU(cudaGetLastError());

    CU(cudaGraphicsUnmapResources(1, &prev_pos_cu_, 0));
    CU(cudaGraphicsUnmapResources(1, &pos_cu_, 0));
  }

 private:
  GLuint prev_pos_vbo_ = 0;
  GLuint pos_vbo_ = 0;

  cudaGraphicsResource* prev_pos_cu_ = nullptr;
  cudaGraphicsResource* pos_cu_ = nullptr;

  Vector2* vel_ = nullptr;
  Scalar* mass_ = nullptr;

  int bodies_count_ = 0;
};

static NBody instance;

}  // namespace cuda_tiled
