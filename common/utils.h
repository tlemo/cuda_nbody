
#pragma once

#include <assert.h>

[[noreturn]] void __checkFailed(const char* expr, int line, const char* source);
[[noreturn]] void __checkFailed(const char* expr,
                                int line,
                                const char* source,
                                const char* message,
                                ...);

[[noreturn]] void __fatal(const char* message, ...);

#define CHECK(x, ...)                                       \
  do {                                                      \
    if (!(x))                                               \
      __checkFailed(#x, __LINE__, __FILE__, ##__VA_ARGS__); \
  } while (false)

#define FATAL(msg, ...) __fatal("\nFATAL: " msg "\n\n", ##__VA_ARGS__)

inline constexpr int CeilDiv(int a, int b) {
  assert(a >= 0);
  assert(b > 0);
  return (a + b - 1) / b;
}
