#include "legged_rl_deploy/policy/policy_factory.h"
#include "legged_rl_deploy/policy_slot.h"
#include "legged_rl_deploy/external/ros_image_tensor_input.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <thread>

using namespace legged_rl_deploy;
namespace {
void check(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
template<class F> void rejects(F f, const std::string& message) {
  bool rejected = false;
  try { f(); } catch (const std::runtime_error&) { rejected = true; }
  check(rejected, message);
}
void near(float a, float b, const std::string& message) {
  check(std::isfinite(a) && std::abs(a-b)<5e-5f, message);
}
class ResetProbe : public IPolicyRunner {
 public: bool fail = false;
 private:
  void loadBackend(const std::string&) override {}
  void runBackend(const std::vector<const float*>& in, const std::vector<float*>& out) override {
    if (fail) throw std::runtime_error("intentional inference failure");
    out[0][0] = in[0][0];
  }
};
void resetTest() {
  auto cfg=YAML::Load(R"(model:
  inputs: [{name: reset, source: policy_reset, dtype: bool, shape: [1]}]
  outputs: [{name: action, target: actions, shape: [1]}]
)");
  ResetProbe r; r.load("",cfg); float value=0;
  r.infer({},&value); near(value,1,"self-check must not consume reset");
  r.infer({},&value); near(value,0,"reset must clear after successful infer");
  r.reset();r.fail=true;rejects([&]{r.infer({},&value);},"expected failure");
  r.fail=false;r.infer({},&value);near(value,1,"failure must not consume reset");
}
void timingTest(const std::shared_ptr<rclcpp::Node>& node) {
  auto cfg=YAML::Load(R"(type: ros_image
topic: /beam_contract/depth
encoding: 32FC1
timeout_sec: 0.2
value_range: [0.0, 1.0]
)");
  RosImageTensorInput image(*node,"external.depth",cfg,{1,1,1,1});
  auto pub=node->create_publisher<sensor_msgs::msg::Image>("/beam_contract/depth",rclcpp::SensorDataQoS());
  for(int i=0;i<100 && pub->get_subscription_count()==0;++i) {
    rclcpp::spin_some(node);std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  check(pub->get_subscription_count()>0,"image subscription discovery");
  auto send=[&](int64_t stamp,float value){
    sensor_msgs::msg::Image msg;msg.header.stamp=rclcpp::Time(stamp);
    msg.width=1;msg.height=1;msg.step=4;msg.encoding="32FC1";msg.data.resize(4);
    std::memcpy(msg.data.data(),&value,4);pub->publish(msg);
    for(int i=0;i<5;++i){rclcpp::spin_some(node);std::this_thread::sleep_for(std::chrono::milliseconds(2));}
  };
  std::vector<float> values;
  const auto stamp=node->now().nanoseconds()-1000000;
  send(stamp,.25f);
  auto first=image.readWithTiming(values);near(values[0],.25f,"image pixel");
  near(first.fresh,1,"first frame fresh");check(first.age>0 && first.age<.2,"capture age");
  near(image.readWithTiming(values).fresh,1,"uncommitted frame remains fresh");
  image.commitFrame(first.stamp_ns);
  near(image.readWithTiming(values).fresh,0,"held frame not fresh");
  image.resetTiming();near(image.readWithTiming(values).fresh,1,"reset forces first frame fresh");
  image.commitFrame(first.stamp_ns);
  const auto next=node->now().nanoseconds()-1000000;send(next,.75f);
  auto second=image.readWithTiming(values);near(values[0],.75f,"new pixels");
  near(second.fresh,1,"new frame fresh");check(second.stamp_ns==next,"pixels/timestamp same snapshot");
  // A callback arriving before commit must not mark its new frame consumed.
  image.commitFrame(first.stamp_ns);near(image.readWithTiming(values).fresh,1,"callback/commit race");
  image.commitFrame(second.stamp_ns);send(next-1000,.5f);
  rejects([&]{image.readWithTiming(values);},"backwards timestamp rejected");
  send(node->now().nanoseconds()-1000000000,.5f);
  rejects([&]{image.readWithTiming(values);},"old capture despite fresh arrival rejected");
  send(node->now().nanoseconds()+1000000000,.5f);
  rejects([&]{image.readWithTiming(values);},"future capture rejected");
  send(0,.5f);rejects([&]{image.readWithTiming(values);},"missing timestamp rejected");
}
void modelsTest(const std::shared_ptr<rclcpp::Node>& node, const std::string& root,
                const YAML::Node& reference) {
  LeggedModel model;
  model.loadConfig(YAML::LoadFile(std::string(std::getenv("WORKSPACE"))+"/src/legged_base/config/g1_29dof-config.yaml"));
  LeggedState state(29,model.jointOrder());
  state.setBaseRotationFromEulerZYX(Eigen::Vector3d::Zero());
  state.setJointPos(Eigen::VectorXd::Zero(29));state.setJointVel(Eigen::VectorXd::Zero(29));
  state.setBaseAngularVelocityB(Eigen::Vector3d::Zero());
  auto pub=node->create_publisher<sensor_msgs::msg::Image>("/g1_sim/beam_depth/normalized",rclcpp::SensorDataQoS());
  for(int v=1;v<=4;++v) {
    const auto name="V"+std::to_string(v);
    auto cfg=YAML::LoadFile(root+"/policies/g1/beam/g1_beam_depth_distillation/"+name+"/config_sim.yaml")["policy"];
    auto runner=makePolicyRunner(cfg["backend"].as<std::string>());
    runner->load(std::string(std::getenv("WORKSPACE"))+"/"+cfg["model_path"].as<std::string>(),cfg);
    std::vector<float> history(5978,.1f),depth(120*212,.5f),action(29);
    float fresh=1,age=0;
    std::vector<RuntimeTensor> in={{"observations",history.data(),history.size()},
      {"external.depth",depth.data(),depth.size()},{"image_fresh.depth",&fresh,1},{"image_age.depth",&age,1}};
    for(int step=0;step<4;++step) {
      if(step==3)runner->reset();fresh=step==1?0.f:1.f;age=step==1?.02f:0.f;
      runner->infer(in,action.data());
      auto expected=reference[name][step]["action"].as<std::vector<float>>();
      auto hidden=reference[name][step]["hidden"].as<std::vector<float>>();
      for(size_t j=0;j<action.size();++j)near(action[j],expected[j],name+" Python/C++ action parity");
      for(size_t j=0;j<hidden.size();++j)near(runner->outputBuffer(1)[j],hidden[j],name+" recurrent parity");
    }
    // Exercise actual PolicySlot image bindings, timing, history and reset wiring.
    PolicySlot slot(name,cfg,model,*node);slot.init();slot.reset(state);
    for(int i=0;i<100 && pub->get_subscription_count()==0;++i){rclcpp::spin_some(node);std::this_thread::sleep_for(std::chrono::milliseconds(10));}
    sensor_msgs::msg::Image msg;msg.header.stamp=node->now();
    msg.width=212;msg.height=120;msg.step=212*4;msg.encoding="32FC1";msg.data.resize(depth.size()*4);
    std::memcpy(msg.data.data(),depth.data(),msg.data.size());pub->publish(msg);
    for(int i=0;i<10;++i){rclcpp::spin_some(node);std::this_thread::sleep_for(std::chrono::milliseconds(2));}
    unitree::common::Gamepad pad;pad.ly=.6f;
    slot.update(state,pad,0,.005);check(slot.hasValidOutput(),name+" slot inference");
    slot.update(state,pad,4,.005);check(slot.hasValidOutput(),name+" held-frame inference");
    slot.reset(state);slot.update(state,pad,0,.005);check(slot.hasValidOutput(),name+" slot reset");
  }
}
}
int main(int argc,char**argv) {
  if(argc<2 || !std::getenv("WORKSPACE")){std::cerr<<"reference YAML and WORKSPACE required\n";return 2;}
  rclcpp::init(argc,argv);int result=0;
  try {
    auto node=std::make_shared<rclcpp::Node>("beam_variant_contract_test");
    resetTest();timingTest(node);
    modelsTest(node,std::string(std::getenv("WORKSPACE"))+"/src/legged_rl_deploy",YAML::LoadFile(argv[1]));
    std::cout<<"PASS: V1--V4 Python/C++ ONNX parity, PolicySlot ROS images, boolean reset, frame timing and timestamp failures\n";
  } catch(const std::exception&e){std::cerr<<e.what()<<'\n';result=1;}
  rclcpp::shutdown();return result;
}
