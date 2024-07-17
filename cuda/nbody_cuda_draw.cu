
// Structure-of-Arrays + Tiled + Direct buffer rendering

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

namespace cuda_direct_draw {

constexpr int kTileSize = 128;

struct UpdateKernelArgs {
  Vector2* prev_pos = nullptr;
  Vector2* pos = nullptr;
  Vector2* vel = nullptr;
  Scalar* mass = nullptr;
  int count = 0;
};

__shared__ Vector2 pos_cache[kTileSize];

__global__ static void UpdateKernel(UpdateKernelArgs args) {
  assert(blockDim.x == kTileSize);

  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  assert(index < args.count);

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

struct RenderKernelArgs {
  uint32_t* raster_buff = nullptr;
  Vector2* pos = nullptr;
  Vector2* vel = nullptr;
  Scalar* mass = nullptr;
  int count = 0;
  int width = 0;
  int height = 0;
};

__device__ uint32_t PackRgb(uint8_t r, uint8_t g, uint8_t b) {
  return (b << 16) | (g << 8) | r;
}

__global__ static void NewFrameKernel(RenderKernelArgs args) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col < args.width && row < args.height) {
    uint32_t& pixel = args.raster_buff[row * args.width + col];
#if 1  // "fireworks"
    pixel =
        PackRgb((pixel >> 1) & 0xff, (pixel >> 9) & 0xff, (pixel >> 17) & 0xff);
#elif 0  // simple motion blur
    pixel =
        PackRgb((pixel >> 1) & 0x7f, (pixel >> 9) & 0x7f, (pixel >> 17) & 0x7f);
#else
    pixel = PackRgb(0, 0, 0);
#endif
  }
}

__global__ static void RenderKernel(RenderKernelArgs args) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < args.count) {
    const auto pos = args.pos[index];
    const int col = (pos.x + kMaxCoord) / (2 * kMaxCoord) * args.width;
    const int row = (pos.y + kMaxCoord) / (2 * kMaxCoord) * args.height;
    if (col >= 0 && col < args.width && row >= 0 && row < args.height) {
      args.raster_buff[row * args.width + col] = PackRgb(255, 255, 255);
    }
  }
}

class NBody : public NBodyPlugin {
 public:
  NBody() : NBodyPlugin("cuda_surface_draw") {}

 private:
  void Free() {
    if (texture_cu_ != nullptr) {
      glBindTexture(GL_TEXTURE_2D, 0);
      CU(cudaGraphicsUnregisterResource(texture_cu_));
      glDeleteTextures(1, &texture_);
      texture_cu_ = nullptr;
      texture_ = 0;
    }

    CU(cudaFree(pos_));
    CU(cudaFree(prev_pos_));
    CU(cudaFree(vel_));
    CU(cudaFree(mass_));
    pos_ = nullptr;
    prev_pos_ = nullptr;
    vel_ = nullptr;
    mass_ = nullptr;

    CU(cudaFree(raster_buff_));
    raster_buff_ = nullptr;

    bodies_count_ = 0;

    width_ = 0;
    height_ = 0;
  }

  void Init(const std::vector<Body>& bodies, int width, int height) final {
    PrintCudaInfo();

    // reset state
    Free();

    width_ = width;
    height_ = height;

    bodies_count_ = CeilDiv(bodies.size(), kTileSize) * kTileSize;
    CHECK(bodies_count_ > 0);

    printf("Tile size: %d\n", kTileSize);
    printf("Rounding up the number of bodies to tile size: %d\n",
           bodies_count_);

    const size_t buffer_size = bodies_count_ * sizeof(Vector2);

    // CUDA buffers
    CU(cudaMallocManaged(&pos_, buffer_size));
    CU(cudaMallocManaged(&prev_pos_, buffer_size));
    CU(cudaMallocManaged(&vel_, buffer_size));
    CU(cudaMallocManaged(&mass_, bodies_count_ * sizeof(Scalar)));
    CU(cudaMallocManaged(&raster_buff_, sizeof(uint32_t) * width_ * height_));

    // OpenGL texture
    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA,
                 width_,
                 height_,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 nullptr);
    CHECK(glGetError() == GL_NO_ERROR);

    // register textures with CUDA
    CU(cudaGraphicsGLRegisterImage(&texture_cu_,
                                   texture_,
                                   GL_TEXTURE_2D,
                                   cudaGraphicsMapFlagsWriteDiscard));

    // copy initial values
    for (int i = 0; i < bodies_count_; ++i) {
      if (i < bodies.size()) {
        pos_[i] = bodies[i].pos;
        vel_[i] = bodies[i].v;
        mass_[i] = bodies[i].mass;
      } else {
        pos_[i] = { 0, 0 };
        vel_[i] = { 0, 0 };
        mass_[i] = kMinBodyMass;
      }
    }
  }

  void Shutdown() final { Free(); }

  void Render() final {
    glBindTexture(GL_TEXTURE_2D, texture_);
    glEnable(GL_TEXTURE_2D);

    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glBegin(GL_QUADS);
    glTexCoord2i(0, 1);
    glVertex2f(-kMaxCoord, -kMaxCoord);
    glTexCoord2i(0, 0);
    glVertex2f(-kMaxCoord, kMaxCoord);
    glTexCoord2i(1, 0);
    glVertex2f(kMaxCoord, kMaxCoord);
    glTexCoord2i(1, 1);
    glVertex2f(kMaxCoord, -kMaxCoord);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
  }

  void Update() final {
    std::swap(pos_, prev_pos_);

    UpdateKernelArgs update_args;
    update_args.prev_pos = prev_pos_;
    update_args.pos = pos_;
    update_args.vel = vel_;
    update_args.mass = mass_;
    update_args.count = bodies_count_;

    const int kUpdateBlockDim = kTileSize;
    const int kUpdateGridDim = CeilDiv(bodies_count_, kUpdateBlockDim);
    UpdateKernel<<<kUpdateGridDim, kUpdateBlockDim>>>(update_args);

    RenderKernelArgs render_args;
    render_args.raster_buff = raster_buff_;
    render_args.pos = pos_;
    render_args.vel = vel_;
    render_args.mass = mass_;
    render_args.count = bodies_count_;
    render_args.width = width_;
    render_args.height = height_;

    const auto kClearBlockDim = dim3(32, 32);
    const auto kClearGridDim = dim3(CeilDiv(width_, kClearBlockDim.x),
                                    CeilDiv(height_, kClearBlockDim.y));
    NewFrameKernel<<<kClearGridDim, kClearBlockDim>>>(render_args);

    const auto kRenderBlockDim = 256;
    const auto kRenderGridDim = CeilDiv(bodies_count_, kRenderBlockDim);
    RenderKernel<<<kRenderGridDim, kRenderBlockDim>>>(render_args);

    // CU(cudaGetLastError());
    // CU(cudaDeviceSynchronize());

    UpdateTexture();
  }

 private:
  void UpdateTexture() {
    CU(cudaGraphicsResourceSetMapFlags(texture_cu_,
                                       cudaGraphicsMapFlagsWriteDiscard));
    CU(cudaGraphicsMapResources(1, &texture_cu_, 0));

    cudaArray* texture_array = nullptr;
    CU(cudaGraphicsSubResourceGetMappedArray(
        &texture_array, texture_cu_, 0, 0));

    const size_t width_bytes = sizeof(uint32_t) * width_;
    CU(cudaMemcpy2DToArray(texture_array,
                           0,
                           0,
                           raster_buff_,
                           width_bytes,
                           width_bytes,
                           height_,
                           cudaMemcpyDeviceToDevice));

    CU(cudaGraphicsUnmapResources(1, &texture_cu_, 0));
  }

 private:
  int width_ = 0;
  int height_ = 0;

  GLuint texture_ = 0;

  // 32bit RGBA linear buffer
  // (updated by the rendering kernel)
  uint32_t* raster_buff_ = nullptr;

  cudaGraphicsResource* texture_cu_ = nullptr;

  Vector2* pos_ = nullptr;
  Vector2* prev_pos_ = nullptr;
  Vector2* vel_ = nullptr;
  Scalar* mass_ = nullptr;

  int bodies_count_ = 0;
};

static NBody instance;

}  // namespace cuda_direct_draw
