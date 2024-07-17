
#pragma once

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <nvtx3/nvToolsExt.h>

#include <cstdlib>

//------------------------------------------------------------------------------

#define CU(stm)                                                            \
  do {                                                                     \
    const cudaError_t err = (stm);                                         \
    if (err != cudaSuccess) {                                              \
      printf("\nCUDA API Error: %d (%s)\n", err, cudaGetErrorString(err)); \
      printf(" > %s [%s:%d]\n\n", #stm, __FILE__, __LINE__);               \
      std::abort();                                                        \
    }                                                                      \
  } while (0)


//------------------------------------------------------------------------------

#define DUMP_CUDA_PROP(prop, format) \
  printf("  ." #prop " : " format "\n", cu_device_prop.prop)

inline void PrintCudaInfo() {
  int cu_device = -1;
  CU(cudaGetDevice(&cu_device));

  cudaDeviceProp cu_device_prop = {};
  CU(cudaGetDeviceProperties(&cu_device_prop, cu_device));

  printf("\nInitializing CUDA ...\n");
  DUMP_CUDA_PROP(name, "%s");
  DUMP_CUDA_PROP(major, "%d");
  DUMP_CUDA_PROP(minor, "%d");
  DUMP_CUDA_PROP(totalGlobalMem, "%zu");
  DUMP_CUDA_PROP(totalConstMem, "%zu");
  DUMP_CUDA_PROP(sharedMemPerBlock, "%zu");
  DUMP_CUDA_PROP(regsPerBlock, "%d");
  DUMP_CUDA_PROP(multiProcessorCount, "%d");
  DUMP_CUDA_PROP(warpSize, "%d");
  DUMP_CUDA_PROP(maxThreadsPerBlock, "%d");
  DUMP_CUDA_PROP(maxThreadsPerMultiProcessor, "%d");
  DUMP_CUDA_PROP(maxThreadsDim[0], "%d");
  DUMP_CUDA_PROP(maxThreadsDim[1], "%d");
  DUMP_CUDA_PROP(maxThreadsDim[2], "%d");
  DUMP_CUDA_PROP(regsPerMultiprocessor, "%d");
  DUMP_CUDA_PROP(maxGridSize[0], "%d");
  DUMP_CUDA_PROP(maxGridSize[1], "%d");
  DUMP_CUDA_PROP(maxGridSize[2], "%d");
  DUMP_CUDA_PROP(clockRate, "%d");
  DUMP_CUDA_PROP(kernelExecTimeoutEnabled, "%d");
  DUMP_CUDA_PROP(canMapHostMemory, "%d");
  DUMP_CUDA_PROP(computeMode, "%d");
  DUMP_CUDA_PROP(unifiedAddressing, "%d");
  DUMP_CUDA_PROP(memoryClockRate, "%d");
  DUMP_CUDA_PROP(memoryBusWidth, "%d");
  DUMP_CUDA_PROP(l2CacheSize, "%d");
  DUMP_CUDA_PROP(managedMemory, "%d");
  DUMP_CUDA_PROP(directManagedMemAccessFromHost, "%d");
  printf("\n");
}

#undef DUMP_CUDA_PROP


//------------------------------------------------------------------------------

class CudaProfilerRange {
 public:
  explicit CudaProfilerRange(const char* name) { nvtxRangePushA(name); }
  ~CudaProfilerRange() { nvtxRangePop(); }
};
