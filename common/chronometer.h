
#pragma once

#include <chrono>
#include <stdio.h>

class Chronometer {
 public:
  using Clock = std::chrono::steady_clock;

 public:
  explicit Chronometer(const char* name, double* elapsed = nullptr)
      : name_(name), elapsed_(elapsed) {
    start_timestamp_ = Clock::now();
  }

  ~Chronometer() {
    const auto finish_timestamp = Clock::now();
    const std::chrono::duration<double> d = finish_timestamp - start_timestamp_;
    const double elapsed = d.count() * 1000.0;
    printf("Elapsed(%s): %.3f ms\n", name_, elapsed);
    if (elapsed_ != nullptr) {
      *elapsed_ = elapsed;
    }
  }

 private:
  Clock::time_point start_timestamp_;
  const char* name_ = nullptr;
  double* elapsed_ = nullptr;
};
