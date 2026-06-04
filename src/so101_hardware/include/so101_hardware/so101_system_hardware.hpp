#ifndef SO101_HARDWARE__SO101_SYSTEM_HARDWARE_HPP_
#define SO101_HARDWARE__SO101_SYSTEM_HARDWARE_HPP_

#include <map>
#include <string>
#include <vector>
#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "ibrobot_msgs/msg/joint_current.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp/publisher.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "SMS_STS.h"

namespace so101_hardware
{
class SO101SystemHardware : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(SO101SystemHardware)

  hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override;
  hardware_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;
  hardware_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::return_type read(const rclcpp::Time & time, const rclcpp::Duration & period) override;
  hardware_interface::return_type write(const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  void publish_currents(const rclcpp::Time & stamp);

  SMS_STS sms_sts_;
  std::string port_;
  std::string calib_file_;
  std::string reset_positions_str_;
  std::vector<double> hw_positions_;
  std::vector<double> hw_velocities_;
  std::vector<double> hw_currents_;
  std::vector<double> hw_commands_;
  std::vector<u8> motor_ids_;
  std::vector<s16> target_positions_;
  std::vector<u16> target_speeds_;
  std::vector<u8> target_accs_;
  std::map<u8, int> homing_offsets_;
  std::map<u8, int> range_mins_;
  std::map<u8, int> range_maxes_;
  std::vector<double> reset_positions_;
  bool has_reset_positions_;
  rclcpp::Node::SharedPtr current_node_;
  rclcpp::Publisher<ibrobot_msgs::msg::JointCurrent>::SharedPtr current_pub_;
};

}  // namespace so101_hardware

#endif  // SO101_HARDWARE__SO101_SYSTEM_HARDWARE_HPP_
