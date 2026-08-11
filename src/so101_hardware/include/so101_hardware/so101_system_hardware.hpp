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

// One decoded Feetech feedback frame for a single motor (raw servo units).
// Promoted to the header so the activation initial-sync helper below can be
// unit-tested without a live serial bus.
struct FeedbackSample
{
  int position{0};
  int speed{0};
  int current{0};
};

// Outcome of the initial feedback sync read performed during on_activate.
// The two failure modes are reported distinctly so the caller can log a
// precise diagnostic before routing through abort_activation.
struct InitialSyncFeedbackOutcome
{
  // True only when every motor returned a full, CRC-valid feedback frame and
  // hw_commands/positions/velocities/currents were fully seeded.
  bool success{false};
  // True when the failure was the sync-read transmit / bus reply (Tx returned
  // <= 0). When false and success is false, the failure was a per-motor Rx.
  bool tx_failed{false};
  // Motor ID whose syncReadPacketRx failed first (valid only when !success &&
  // !tx_failed). Reported as-is in the activation abort log.
  u8 failed_motor_id{0};
};

// Seed hw_commands/positions/velocities/currents from the first sync-read
// round after torque is enabled. Fail-closed: returns success=false if the
// sync-read transmit fails (do_sync_tx returns <= 0) or if ANY motor's Rx
// fails (do_sync_rx returns false), so on_activate can route through
// abort_activation (torque off / EPROM relock / port close) instead of
// dismissing the rollback guard with partially-initialised state.
//
// This helper is hardware-agnostic: it takes the SMS_STS sync-read operations
// as callbacks, so unit tests can inject Tx/Rx failures without a real bus.
// Note: SMS_STS::syncReadBegin returns void (verified in SCS.h) and only
// allocates the SDK receive buffer, so it is NOT a fail-closed gate; the gate
// is the Tx return value checked here.
InitialSyncFeedbackOutcome perform_initial_sync_feedback(
  const std::vector<u8> & motor_ids,
  bool has_reset_positions,
  const std::vector<double> & reset_positions,
  double ticks_per_rad,
  double current_raw_to_ampere,
  const std::function<int()> & do_sync_tx,
  const std::function<bool(u8, FeedbackSample &)> & do_sync_rx,
  std::vector<double> & hw_commands,
  std::vector<double> & hw_positions,
  std::vector<double> & hw_velocities,
  std::vector<double> & hw_currents);
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
