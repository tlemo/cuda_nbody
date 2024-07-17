
#pragma once

#include <assert.h>
#include <chrono>
#include <algorithm>

// A simple event-per-second rate tracker (ex. FPS)
class RateTracker {
  using Clock = std::chrono::steady_clock;

  static constexpr int kMaxTrackedUpdates = 30;
  static constexpr int kMinAggregatedUpdates = 5;

 public:
  RateTracker() { Reset(); }

  void Reset() {
    next_update_ = 0;
    updates_count_ = 0;
    start_timestamp_ = Clock::now();
    last_report_ = start_timestamp_;
  }

  void Update() {
    auto timestamp = Clock::now();
    assert(next_update_ < kMaxTrackedUpdates);
    update_times_[next_update_++] = timestamp;
    if (next_update_ == kMaxTrackedUpdates) {
      next_update_ = 0;
    }
    ++updates_count_;
  }

  bool ShouldReport(double interval_sec) {
    assert(interval_sec >= 0.0);
    const std::chrono::duration<double> delta = Clock::now() - last_report_;
    if (delta.count() >= interval_sec) {
      last_report_ = Clock::now();
      return true;
    }
    return false;
  }

  double current_rate() const {
    const int tracked_updates = std::min(updates_count_, kMaxTrackedUpdates);

    if (tracked_updates < kMinAggregatedUpdates) {
      return 0;
    }

    const int last_update =
        (kMaxTrackedUpdates + next_update_ - 1) % kMaxTrackedUpdates;
    const int oldest_update =
        (kMaxTrackedUpdates + next_update_ - tracked_updates) %
        kMaxTrackedUpdates;

    const std::chrono::duration<double> delta =
        update_times_[last_update] - update_times_[oldest_update];
    const double seconds = delta.count();
    return tracked_updates / seconds;
  }

  double average_rate() const {
    if (updates_count_ > 0) {
      const std::chrono::duration<double> delta =
          Clock::now() - start_timestamp_;
      const double seconds = delta.count();
      return updates_count_ / seconds;
    } else {
      return 0.0;
    }
  }

  int updates_count() const { return updates_count_; }

 private:
  Clock::time_point start_timestamp_;
  Clock::time_point update_times_[kMaxTrackedUpdates];
  int next_update_ = 0;
  int updates_count_ = 0;
  Clock::time_point last_report_;
};
