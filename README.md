
# NBody simulation examples

Various N-body implementations, demonstrating multiple CPU and CUDA optimizations

## Building from source

#### Prerequisites

1. A C++ 17 compiler (GCC, Clang, MSVC)
2. CUDA SDK 11.0 or newer
3. CMake 3.25 or newer

#### Getting the Source Code

This source tree uses [Git submodules][5] for third party libraries,
and `--recursive` is required when cloning:

```shell
git clone --recursive <repository>
```

#### Building on Windows

When building on Windows with MSVC, the CMake configuration must be done from
a MSVC command prompt (ex. run `vcvarsall.bat`)

#### Building on Linux

A few additional system tools and libraries may be required. The CMake 
configuration phase might fail if they are missing, and it would log 
instructions on how to install them.

For example, on a recent Ubuntu / Debian system, use the following commands:

```
sudo apt-get install curl zip unzip tar
sudo apt-get install libxmu-dev libxi-dev libgl-dev
sudo apt install libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev
```

#### Building with CMake

> Tip: CMake uses the `CUDA_PATH` environment variable. Make sure it is set
> and points to the right location before running `cmake`.

This project uses CMake, so the general build process looks like:

```shell
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8 # replace 8 with the number of cores
```

## Running the N-body simulation

#### Running on Linux + Wayland 

If the app exits with the following initialization error:

```
CUDA API Error: 999 (unknown error)
 > cudaGraphicsGLRegisterImage(...)
```

... then try setting the following environment variables first:

```
export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
```

## Math

This 2D N-body simulation is based on a simple Verlet integration using 
the following equations:

$F = G \frac {m_1 m_2} {r^2}$

$a = \frac{F}{m} = G \frac{m_2}{r^2}$

$\vec{a} = \frac{\vec{r}}{|r|}a = G \frac{\vec{r} m_2}{r^3}$

## References

- [N-body problem](https://en.wikipedia.org/wiki/N-body_problem)
- [Fast N-Body Simulation with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda)
