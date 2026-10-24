#include "legged_rl_deploy/policy_slot.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace legged_rl_deploy {
namespace {
void check(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
void near(float a, float b, const std::string& message) {
  check(std::isfinite(a) && std::abs(a - b) < 3e-5f,
        message + ": got " + std::to_string(a) + ", expected " + std::to_string(b));
}
class ConstantRunner : public IPolicyRunner {
  void loadBackend(const std::string&) override {}
  void runBackend(const std::vector<const float*>&, const std::vector<float*>& out) override {
    for (size_t j = 0; j < actionDim(); ++j) out[0][j] = j % 2 ? -100.0f : 100.0f;
  }
};
}  // namespace

struct PolicySlotContractTest {
  static void run(rclcpp::Node& node) {
    const char* workspace = std::getenv("WORKSPACE");
    check(workspace != nullptr, "Set WORKSPACE to unitree_ws");
    const std::string root = std::string(workspace) + "/src/legged_rl_deploy/";
    auto cfg = YAML::LoadFile(root + "policies/g1/velocity/GRU61Delay/config.yaml")["policy"];
    LeggedModel model;
    model.loadConfig(YAML::LoadFile(std::string(workspace) + "/src/legged_base/config/g1_29dof-config.yaml"));
    LeggedState state(29, model.jointOrder());
    state.setBaseRotationFromEulerZYX(Eigen::Vector3d::Zero());
    state.setJointPos(Eigen::VectorXd::Zero(29));
    state.setJointVel(Eigen::VectorXd::Zero(29));
    state.setBaseAngularVelocityB(Eigen::Vector3d::Zero());
    unitree::common::Gamepad pad;
    pad.ly = 1.0f; pad.lx = 0.5f; pad.rx = 0.5f;
    PolicySlot slot("narrow_test", cfg, model, node);
    slot.init();  // Loads and validates the real ONNX export; no hardware adapter.
    slot.reset(state);
    slot.update(state, pad, 0, .005);
    for (float v : slot.outputBuf()) check(std::isfinite(v), "Real ONNX inference");
    check(slot.fixedJoints().empty(), "Original fixed-joint configuration unchanged");
    const auto first_real_output = slot.outputBuf();
    slot.reset(state);
    slot.update(state, pad, 0, .005);
    for (size_t j = 0; j < first_real_output.size(); ++j)
      near(slot.outputBuf()[j], first_real_output[j], "Real ONNX reset parity");

    auto fake = std::make_unique<ConstantRunner>();
    fake->load("", cfg);
    slot.policy_runner_ = std::move(fake);
    slot.reset(state);
    const auto lower = cfg["actions"]["AppliedActionClip"]["min"].as<std::vector<float>>();
    const auto upper = cfg["actions"]["AppliedActionClip"]["max"].as<std::vector<float>>();
    const auto scale = cfg["actions"]["JointPositionAction"]["scale"].as<std::vector<float>>();
    const auto offset = cfg["actions"]["JointPositionAction"]["offset"].as<std::vector<float>>();
    std::vector<std::vector<float>> frames;
    float command = 0;
    double phase = 0, period = 1.1;
    const int starts[] = {0, 3, 6, 9, 11, 23, 35};
    const int sizes[] = {3, 3, 3, 2, 12, 12, 12};
    // Stand, accelerate, cruise, stop (phase freezes), then restart.
    for (size_t tick = 0; tick < 420; ++tick) {
      pad.ly = tick < 250 || tick >= 350 ? 1.0f : 0.0f;
      // Include the exact deadband boundary and a below-threshold command.
      if (tick == 5) pad.ly = .1f;
      if (tick == 6) pad.ly = .05f;
      command = std::clamp(pad.ly, 0.f, .8f);
      if (command <= .1f) command = 0;
      const bool active = command > .1f;
      if (tick > 0 && active) phase = std::fmod(phase + .02 / .9, 1.0);
      const float gyro = float(tick % 25) - 12.f;
      state.setBaseAngularVelocityB(Eigen::Vector3d(gyro, 0, 0));
      Eigen::VectorXd q = Eigen::VectorXd::Constant(29, 999.0);
      Eigen::VectorXd dq = Eigen::VectorXd::Constant(29, 999.0);
      for (int j = 0; j < 12; ++j) {
        q[j] = offset[j] + .01 * float(tick % 250) + .01 * j;
        dq[j] = float(tick % 80) - 40.f + j;
      }
      state.setJointPos(q); state.setJointVel(dq);
      std::vector<float> frame{std::clamp(gyro, -10.f, 10.f) * .25f, 0, 0,
                               0, 0, -1, command, 0, 0,
                               active ? float(std::sin(2*M_PI*phase)) : 0.f,
                               active ? float(std::cos(2*M_PI*phase)) : 0.f};
      for (int j = 0; j < 12; ++j) frame.push_back(std::clamp(float(q[j])-offset[j], -2.f, 2.f));
      for (int j = 0; j < 12; ++j) frame.push_back(std::clamp(float(dq[j]), -30.f, 30.f)*.05f);
      for (int j = 0; j < 12; ++j) frame.push_back(tick == 0 ? 0.f : (j%2 ? lower[j] : upper[j]));
      frames.push_back(frame);
      slot.update(state, pad, tick*4, .005);
      size_t out = 0;
      for (int term = 0; term < 7; ++term) {
        for (int lag = 60; lag >= 0; --lag) {
          const auto& f = frames[tick >= size_t(lag) ? tick-lag : 0];
          for (int j = 0; j < sizes[term]; ++j) {
            near(slot.input_buf_.at(out++), f[starts[term]+j], "History tick " + std::to_string(tick));
          }
        }
      }
      check(out == 2867, "Term-major input size");
      for (int j = 0; j < 12; ++j) {
        const float applied = j%2 ? lower[j] : upper[j];
        near(slot.last_action_[j], applied, "Previous clipped raw action");
        near(slot.outputBuf()[j], offset[j]+scale[j]*applied, "Joint target");
      }
      for (int substep = 1; substep < 4; ++substep) slot.update(state, pad, tick*4+substep, .005);
      check(slot.policy_ticks_ == tick+1, "History updates once per policy tick");
    }
    slot.reset(state);
    slot.update(state, pad, 1680, .005);
    near(slot.narrow_phase_, 0.f, "Reset phase independent of global clock");
    for (size_t i = 2135; i < 2867; ++i) near(slot.input_buf_[i], 0.f, "Reset previous action history");
    near(slot.velocity_command_[0], .8f, "Effective command on reset");

    std::cout << "PASS: real ONNX inference, all 2867 history entries over 420 ticks, clipping, phase, reset, decimation\n";
  }
};
}  // namespace legged_rl_deploy

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int result = 0;
  try {
    auto node = std::make_shared<rclcpp::Node>("policy_contract_test");
    legged_rl_deploy::PolicySlotContractTest::run(*node);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n'; result = 1;
  }
  rclcpp::shutdown();
  return result;
}
