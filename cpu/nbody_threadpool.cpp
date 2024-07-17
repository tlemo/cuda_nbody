
#include <common/nbody_plugin.h>
#include <common/utils.h>
#include <common/math_2d.h>

#include <GL/glew.h>
#include <GL/gl.h>

#include <stdio.h>
#include <utility>
#include <vector>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <math.h>

class ThreadPoolNBody : public NBodyPlugin {
  struct WorkItem {
    int begin_index = -1;
    int end_index = -1;
  };

 public:
  ThreadPoolNBody() : NBodyPlugin("cpu_threadpool") {}

 private:
  void Init(const std::vector<Body>& bodies, int, int) final {
    bodies_ = bodies;
    prev_bodies_.resize(bodies_.size());
    CreateThreadPool();
  }

  void Shutdown() final {
    {
      std::unique_lock<std::mutex> guard(work_queue_lock_);
      shutdown_ = true;
      work_queue_cv_.notify_all();
    }

    for (auto& thread : worker_threads_) {
      thread.join();
    }
  }

  void Render() final {
    glColor3f(0.5, 1.0, 0.9);
    glPointSize(1.0);
    glBegin(GL_POINTS);
    for (const auto& body : bodies_) {
      glVertex2f(body.pos.x, body.pos.y);
    }
    glEnd();
  }

  void Update() final {
    std::unique_lock<std::mutex> guard(work_queue_lock_);
    CHECK(next_work_item_ == work_queue_.size());
    CHECK(work_left_ == 0);

    const int workers_count = worker_threads_.size();
    CHECK(workers_count > 0);

    std::swap(bodies_, prev_bodies_);

    // setup work items
    size_t current = 0;
    const size_t chunk_size = bodies_.size() / workers_count;
    const size_t rem_size = bodies_.size() % workers_count;
    work_queue_.resize(workers_count);
    for (int i = 0; i < workers_count; ++i) {
      auto& work_item = work_queue_[i];
      const size_t work_item_size = i < rem_size ? chunk_size + 1 : chunk_size;
      work_item.begin_index = current;
      work_item.end_index = current + work_item_size;
      current = work_item.end_index;
    }
    CHECK(current == bodies_.size());
    next_work_item_ = 0;
    work_left_ = work_queue_.size();

    // wake up workers
    work_queue_cv_.notify_all();

    // wait for the work to finish
    while (work_left_ > 0) {
      results_cv_.wait(guard);
    }
  }

  void UpdateBody(int index) {
    Body body = prev_bodies_[index];
    Vector2 acc = { 0.0f, 0.0f };
    for (const auto& other : prev_bodies_) {
      const Vector2 r = other.pos - body.pos;
      const Scalar dist_squared = length_squared(r) + kSofteningFactor;
      const Scalar inv_dist = 1.0f / sqrtf(dist_squared);
      const Scalar inv_dist_cube = inv_dist * inv_dist * inv_dist;
      const Scalar s = other.mass * inv_dist_cube;
      acc = acc + r * s;
    }
    body.v = body.v + acc * kTimeStep;
    body.v = body.v * kDampingFactor;
    body.pos = body.pos + body.v * kTimeStep;
    bodies_[index] = body;
  }

  void WorkerThread() {
    WorkItem work_item;

    for (;;) {
      {
        std::unique_lock<std::mutex> guard(work_queue_lock_);

        // report previous iteration completion
        if (work_item.begin_index != work_item.end_index) {
          CHECK(work_left_ > 0);
          if (--work_left_ == 0) {
            results_cv_.notify_all();
          }
        }

        // acquire new work
        while (!shutdown_ && next_work_item_ == work_queue_.size()) {
          work_queue_cv_.wait(guard);
        }
        if (shutdown_) {
          return;
        }
        CHECK(!work_queue_.empty());
        CHECK(next_work_item_ < work_queue_.size());
        CHECK(work_left_ > 0);
        work_item = work_queue_[next_work_item_++];
      }

      // perform the work
      CHECK(work_item.begin_index < work_item.end_index);
      for (int i = work_item.begin_index; i < work_item.end_index; ++i) {
        UpdateBody(i);
      }
    }
  }

  void CreateThreadPool() {
    const int cpu_count = std::thread::hardware_concurrency();
    printf("\nCreating thread pool, detected %d core(s)...\n", cpu_count);
    for (int i = 0; i < cpu_count; ++i) {
      worker_threads_.emplace_back(&ThreadPoolNBody::WorkerThread, this);
    }
  }

 private:
  std::vector<Body> prev_bodies_;
  std::vector<Body> bodies_;

  std::vector<std::thread> worker_threads_;

  std::vector<WorkItem> work_queue_;
  size_t next_work_item_ = 0;
  int work_left_ = 0;
  bool shutdown_ = false;
  std::mutex work_queue_lock_;
  std::condition_variable work_queue_cv_;
  std::condition_variable results_cv_;
};

static ThreadPoolNBody instance;
