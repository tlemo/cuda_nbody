
#include "nbody_plugin.h"

#include <common/utils.h>

#include <map>
#include <string>

static std::map<std::string, NBodyPlugin*> g_plugins;

NBodyPlugin::NBodyPlugin(const std::string& name) {
  RegisterPlugin(this, name);
}

void RegisterPlugin(NBodyPlugin* plugin, const std::string& name) {
  CHECK(g_plugins.insert({ name, plugin }).second);
}

NBodyPlugin* SelectPlugin(const std::string& name) {
  printf("Implementation plugin: %s\n", name.c_str());
  auto it = g_plugins.find(name);
  CHECK(it != g_plugins.end());
  return it->second;
}

std::vector<std::string> AvailablePlugins() {
  std::vector<std::string> plugin_names;
  for (const auto& kv : g_plugins) {
    plugin_names.push_back(kv.first);
  }
  return plugin_names;
}
