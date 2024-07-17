
#include <common/rate_tracker.h>
#include <common/utils.h>
#include <common/math_2d.h>
#include <common/nbody_plugin.h>

#include <GL/glew.h>

#include <GLFW/glfw3.h>

#include <GL/gl.h>
#include <GL/glu.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <argparse/argparse.hpp>

#include <stdio.h>
#include <cstdlib>
#include <random>
#include <vector>
#include <stdexcept>

constexpr int kWidth = 1024;
constexpr int kHeight = 1024;

constexpr int kBodiesCount = 25000;
constexpr bool kInitialVelocities = true;

auto GenerateRandomBodies() {
  printf("Generating random initial configuration...\n");
  std::vector<Body> bodies(kBodiesCount);
  std::random_device rd;
  std::default_random_engine rnd(rd());
  std::uniform_real_distribution<Scalar> dist_coord(-kMaxCoord, kMaxCoord);
  std::uniform_real_distribution<Scalar> dist_mass(kMinBodyMass, kMaxBodyMass);
  for (auto& body : bodies) {
    body.pos.x = dist_coord(rnd);
    body.pos.y = dist_coord(rnd);
    body.mass = dist_mass(rnd);
    if (kInitialVelocities) {
      body.v.x = body.pos.y / -50;
      body.v.y = body.pos.x / 50;
    }
  }
  return bodies;
}

void ResizeCallback(GLFWwindow* window, int w, int h) {
  glViewport(0, 0, w, h);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(-kMaxCoord, kMaxCoord, -kMaxCoord, kMaxCoord);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

void KeyboardCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (action != GLFW_PRESS) {
    return;
  }

  static int windowed_xpos = 0;
  static int windowed_ypos = 0;
  static int windowed_width = 0;
  static int windowed_height = 0;

  if (key == GLFW_KEY_ESCAPE && mods == 0)
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  if ((key == GLFW_KEY_ENTER && mods == GLFW_MOD_ALT) ||
      (key == GLFW_KEY_F11 && mods == GLFW_MOD_ALT)) {
    if (glfwGetWindowMonitor(window)) {
      glfwSetWindowMonitor(
          window, NULL, windowed_xpos, windowed_ypos, windowed_width, windowed_height, 0);
    } else {
      if (GLFWmonitor* monitor = glfwGetPrimaryMonitor()) {
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        glfwGetWindowPos(window, &windowed_xpos, &windowed_ypos);
        glfwGetWindowSize(window, &windowed_width, &windowed_height);
        glfwSetWindowMonitor(
            window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
      }
    }
  }
}

void ImguiInit(GLFWwindow* window) {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init();

  ImGui::StyleColorsDark();
}

void ImguiShutdown() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}

void ImguiNewFrame() {
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
}

void ImguiRender() {
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void ImguiDefineUi(double fps) {
  ImGui::Begin("Stats");
  ImGui::Text("FPS: %.2f\n", fps);
  ImGui::End();
}

int main(int argc, const char* argv[]) {
  argparse::ArgumentParser args("nbody");

  auto& plugin_arg = args.add_argument("--plugin").required();
  std::string plugin_arg_help = "Implementation plugin, valid options: ";
  for (const auto& name : AvailablePlugins()) {
    plugin_arg.add_choice(name);
    plugin_arg_help += "\n - " + name;
  }
  plugin_arg.default_value("cuda");
  plugin_arg.help(plugin_arg_help);

  try {
    args.parse_args(argc, argv);
  } catch (const std::runtime_error& e) {
    std::cerr << "\n" << e.what() << "\n\n";
    std::exit(1);
  }

  CHECK(glfwInit(), "Failed to initialize GLFW");

  GLFWwindow* window =
      glfwCreateWindow(kWidth, kHeight, "N-body simulation", nullptr, nullptr);
  CHECK(window != nullptr, "Failed to create main window");

  glfwSetFramebufferSizeCallback(window, ResizeCallback);
  glfwSetKeyCallback(window, KeyboardCallback);

  glfwMakeContextCurrent(window);

  CHECK(glewInit() == GLEW_OK, "Failed to initialize GLEW");

  glfwSwapInterval(0);

  int width = 0;
  int height = 0;
  glfwGetFramebufferSize(window, &width, &height);
  ResizeCallback(window, width, height);

  ImguiInit(window);

  RateTracker fps_tracker;

  auto plugin = SelectPlugin(args.get<std::string>("--plugin"));

  plugin->Init(GenerateRandomBodies(), width, height);

  while (!glfwWindowShouldClose(window)) {
    glClearColor(0.1, 0.1, 0.2, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    ImguiNewFrame();

    plugin->Render();
    plugin->Update();

    ImguiDefineUi(fps_tracker.current_rate());
    ImguiRender();

    glfwSwapBuffers(window);

    fps_tracker.Update();

    // Report FPS every 1.0 seconds
    if (fps_tracker.ShouldReport(1.0)) {
      printf("FPS: %.3f (avg: %.3f)\n",
             fps_tracker.current_rate(),
             fps_tracker.average_rate());
    }

    glfwPollEvents();
  }

  plugin->Shutdown();

  ImguiShutdown();

  glfwDestroyWindow(window);
  glfwTerminate();
}
