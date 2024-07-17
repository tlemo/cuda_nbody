
#pragma once

#ifdef __CUDA_ARCH__
#define CU_D __device__
#else
#define CU_D
#endif  // __CUDA_ARCH__

using Scalar = float;

struct Vector2 {
  Scalar x;
  Scalar y;
};

CU_D inline Vector2 operator-(const Vector2& a, const Vector2& b) {
  return Vector2{ a.x - b.x, a.y - b.y };
}

CU_D inline Vector2 operator+(const Vector2& a, const Vector2& b) {
  return Vector2{ a.x + b.x, a.y + b.y };
}

CU_D inline Scalar length_squared(const Vector2& v) {
  return v.x * v.x + v.y * v.y;
}

CU_D inline Vector2 operator*(const Vector2& v, Scalar s) {
  return Vector2{ v.x * s, v.y * s };
}
