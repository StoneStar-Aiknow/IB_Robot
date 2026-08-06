#ifndef SO101_HARDWARE__SO101_SYSTEM_HARDWARE_HPP_
#define SO101_HARDWARE__SO101_SYSTEM_HARDWARE_HPP_

#include "SMS_STS.h"
#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "ibrobot_msgs/msg/joint_current.hpp"
#include "rclcpp/macros.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp/publisher.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace so101_hardware
{
namespace detail
{
bool disable_torque_on_abort(
  const std::vector<u8> & motor_ids,
  const std::function<int(u8, u8)> & enable_torque,
  int retry_count = 3);

// Outcome of an activation rollback. The two success flags are reported
// independently so callers can surface EPROM relock failures distinctly from
// torque-disable failures while keeping torque-off fail-closed semantics.
struct ActivationRollbackResult
{
  // True only when every motor's torque was disabled.
  bool torque_disabled_all{true};
  // True only when every motor left unlocked was successfully relocked.
  bool eprom_relocked_all{true};
  // Motor IDs whose EPROM could not be relocked (unlocked at abort time).
  std::vector<u8> relock_failures;
};

// Roll back a partial activation: fail-closed torque disable for every motor,
// followed by a best-effort EPROM relock for motors tracked as unlocked.
// EPROM relock runs before the caller closes the serial port.
ActivationRollbackResult rollback_activation(
  const std::vector<u8> & motor_ids, const std::set<u8> & unlocked_motors,
  const std::function<int(u8, u8)> & enable_torque,
  const std::function<int(u8)> & lock_eprom, int retry_count = 3);
} // namespace detail

class SafeSMSSTS : public SMS_STS
{
protected:
  int readSCS(unsigned char * data, int length) override;
  int readSCS(
    unsigned char * data, int length,
    unsigned long timeout_ms) override;

private:
  int read_with_timeout(
    unsigned char * data, int length,
    unsigned long timeout_ms);
};

class SO101SystemHardware : public hardware_interface::SystemInterface
{
public:
  RCLCPP_SHARED_PTR_DEFINITIONS(SO101SystemHardware)

  hardware_interface::CallbackReturn
  on_init(const hardware_interface::HardwareInfo & info) override;
  hardware_interface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State & previous_state) override;
  std::vector<hardware_interface::StateInterface>
  export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface>
  export_command_interfaces() override;
  hardware_interface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::return_type read(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;
  hardware_interface::return_type
  write(const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  void publish_currents(const rclcpp::Time & stamp);

  SafeSMSSTS sms_sts_;
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

} // namespace so101_hardware

#endif // SO101_HARDWARE__SO101_SYSTEM_HARDWARE_HPP_
