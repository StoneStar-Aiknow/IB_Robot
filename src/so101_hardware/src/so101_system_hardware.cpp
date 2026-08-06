#include "so101_hardware/so101_system_hardware.hpp"
#include "SMS_STS.h"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"
#include <cerrno>
#include <chrono>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <utility>

namespace so101_hardware
{
namespace detail
{
class ActivationRollbackGuard
{
public:
  explicit ActivationRollbackGuard(std::function<void()> cleanup)
  : cleanup_(std::move(cleanup)) {}
  ~ActivationRollbackGuard()
  {
    if (active_) {
      cleanup_();
    }
  }
  void dismiss() {active_ = false;}

private:
  std::function<void()> cleanup_;
  bool active_{true};
};

bool disable_torque_on_abort(
  const std::vector<u8> & motor_ids,
  const std::function<int(u8, u8)> & enable_torque,
  int retry_count)
{
  bool all_disabled = true;
  for (auto it = motor_ids.rbegin(); it != motor_ids.rend(); ++it) {
    bool disabled = false;
    for (int attempt = 0; attempt < retry_count; ++attempt) {
      if (enable_torque(*it, 0) != 0) {
        disabled = true;
        break;
      }
      usleep(10000);
    }
    all_disabled = all_disabled && disabled;
  }
  return all_disabled;
}

ActivationRollbackResult
rollback_activation(
  const std::vector<u8> & motor_ids,
  const std::set<u8> & unlocked_motors,
  const std::function<int(u8, u8)> & enable_torque,
  const std::function<int(u8)> & lock_eprom, int retry_count)
{
  ActivationRollbackResult result;
  // Fail-closed: torque OFF is the safe default for every motor, independent
  // of EPROM state. This is attempted first so that relocking (which some
  // Feetech servos only accept while torque is off) is more likely to succeed.
  result.torque_disabled_all =
    disable_torque_on_abort(motor_ids, enable_torque, retry_count);

  // Relock EPROM for any motor still unlocked at abort time, before the caller
  // closes the serial port. Relock is best-effort and reported distinctly so a
  // relock failure is never masked by (or conflated with) a torque outcome.
  for (auto it = motor_ids.rbegin(); it != motor_ids.rend(); ++it) {
    const u8 id = *it;
    if (unlocked_motors.count(id) == 0) {
      continue;
    }
    bool relocked = false;
    for (int attempt = 0; attempt < retry_count; ++attempt) {
      if (lock_eprom(id) != 0) {
        relocked = true;
        break;
      }
      usleep(10000);
    }
    if (!relocked) {
      result.eprom_relocked_all = false;
      result.relock_failures.push_back(id);
    }
  }
  return result;
}

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
  std::vector<double> & hw_currents)
{
  InitialSyncFeedbackOutcome outcome;

  // Fail-closed gate 1: the sync-read transmit + bus reply. syncReadPacketTx
  // returns the number of bytes received into the SDK buffer; <= 0 means no
  // reply (timeout) or a write failure, so there is nothing to decode and the
  // whole activation must abort.
  if (do_sync_tx() <= 0) {
    outcome.success = false;
    outcome.tx_failed = true;
    return outcome;
  }

  // Fail-closed gate 2: EVERY motor must return a full, CRC-valid packet
  // before we seed any state. A single failing motor aborts activation, so we
  // never dismiss the rollback guard with partially-initialised
  // hw_commands/positions/velocities/currents (the values written before the
  // failure are discarded because the caller returns ERROR).
  // Centre tick of the 12-bit (4096-count) STS3215 position range.
  constexpr double kCenterTicks = 2048.0;
  for (size_t i = 0; i < motor_ids.size(); ++i) {
    const u8 id = motor_ids[i];
    FeedbackSample sample{};
    if (!do_sync_rx(id, sample)) {
      outcome.success = false;
      outcome.tx_failed = false;
      outcome.failed_motor_id = id;
      return outcome;
    }

    const double rad =
      (static_cast<double>(sample.position) - kCenterTicks) / ticks_per_rad;
    hw_positions[i] = rad;
    hw_velocities[i] = static_cast<double>(sample.speed) / ticks_per_rad;
    hw_currents[i] = static_cast<double>(sample.current) * current_raw_to_ampere;
    // Hold the current pose unless an explicit reset pose was configured.
    hw_commands[i] = has_reset_positions ? reset_positions[i] : rad;
  }

  outcome.success = true;
  return outcome;
}
} // namespace detail

int SafeSMSSTS::readSCS(unsigned char * data, int length)
{
  return read_with_timeout(data, length, IOTimeOut);
}

int SafeSMSSTS::readSCS(
  unsigned char * data, int length,
  unsigned long timeout_ms)
{
  return read_with_timeout(data, length, timeout_ms);
}

int SafeSMSSTS::read_with_timeout(
  unsigned char * data, int length,
  unsigned long timeout_ms)
{
  if (data == nullptr || length < 0 || fd < 0) {
    return -1;
  }
  if (length == 0) {
    return 0;
  }

  const auto deadline =
    std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  int received = 0;
  while (received < length) {
    const auto remaining = deadline - std::chrono::steady_clock::now();
    if (remaining <= std::chrono::steady_clock::duration::zero()) {
      return received;
    }
    const auto remaining_us =
      std::chrono::duration_cast<std::chrono::microseconds>(remaining)
      .count();
    timeval timeout{};
    timeout.tv_sec = remaining_us / 1000000;
    timeout.tv_usec = remaining_us % 1000000;
    fd_set read_set;
    FD_ZERO(&read_set);
    FD_SET(fd, &read_set);

    const int selected = select(fd + 1, &read_set, nullptr, nullptr, &timeout);
    if (selected == 0) {
      return received;
    }
    if (selected < 0) {
      if (errno == EINTR) {
        continue;
      }
      return received > 0 ? received : -1;
    }

    const ssize_t count =
      ::read(fd, data + received, static_cast<size_t>(length - received));
    if (count > 0) {
      received += static_cast<int>(count);
      continue;
    }
    if (count == 0) {
      return received;
    }
    if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
      return received > 0 ? received : -1;
    }
  }
  return received;
}

namespace
{

constexpr u8 FEEDBACK_START_ADDR = SMS_STS_PRESENT_POSITION_L;
constexpr u8 FEEDBACK_READ_LEN =
  SMS_STS_PRESENT_CURRENT_H - SMS_STS_PRESENT_POSITION_L + 1;
constexpr u32 FEEDBACK_READ_TIMEOUT_MS = 20;
constexpr double CURRENT_RAW_TO_AMPERE = 0.0065;

// Decode a single motor's feedback frame from the SDK packet buffer that
// syncReadPacketRx just populated. FeedbackSample itself lives in detail
// (header) so the initial-sync helper can be unit-tested.
detail::FeedbackSample parse_feedback_packet(SMS_STS & sms_sts)
{
  detail::FeedbackSample sample;
  // syncReadPacketRx() must have just populated the SDK packet/index buffers.
  sample.position = sms_sts.syncReadRxPacketToWrod();
  // syncReadRxPacketToWrod(15) decodes 15-bit signed values per Feetech SDK
  // convention; the SDK handles sign extension internally.
  sample.speed = sms_sts.syncReadRxPacketToWrod(15);
  sms_sts.syncReadRxPacketIndex =
    SMS_STS_PRESENT_CURRENT_L - FEEDBACK_START_ADDR;
  sample.current = sms_sts.syncReadRxPacketToWrod(15);
  return sample;
}

} // namespace

hardware_interface::CallbackReturn
SO101SystemHardware::on_init(const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) !=
    hardware_interface::CallbackReturn::SUCCESS)
  {
    return hardware_interface::CallbackReturn::ERROR;
  }

  port_ = info_.hardware_parameters["port"];
  calib_file_ = info_.hardware_parameters["calib_file"];
  reset_positions_str_ = info_.hardware_parameters["reset_positions"];

  hw_positions_.resize(info_.joints.size(), 0.0);
  hw_velocities_.resize(info_.joints.size(), 0.0);
  hw_currents_.resize(info_.joints.size(), 0.0);
  hw_commands_.resize(info_.joints.size(), 0.0);
  motor_ids_.resize(info_.joints.size());
  target_positions_.resize(info_.joints.size(), 0);
  target_speeds_.resize(info_.joints.size(), 0);
  target_accs_.resize(info_.joints.size(), 0);
  reset_positions_.resize(info_.joints.size(), 0.0);
  has_reset_positions_ = false;

  try {
    if (!reset_positions_str_.empty() && reset_positions_str_ != "''" &&
      reset_positions_str_ != "\"\"")
    {
      auto reset_json = nlohmann::json::parse(reset_positions_str_);
      for (size_t i = 0; i < info_.joints.size(); i++) {
        std::string joint_name = info_.joints[i].name;
        if (reset_json.contains(joint_name)) {
          reset_positions_[i] = reset_json[joint_name];
          has_reset_positions_ = true;
        }
      }
      if (has_reset_positions_) {
        RCLCPP_INFO(
          rclcpp::get_logger("SO101SystemHardware"),
          "Loaded reset positions from config");
      }
    }
  } catch (const std::exception & e) {
    RCLCPP_WARN(
      rclcpp::get_logger("SO101SystemHardware"),
      "Failed to parse reset_positions: %s", e.what());
  }

  for (size_t i = 0; i < info_.joints.size(); i++) {
    motor_ids_[i] = std::stoi(info_.joints[i].parameters.at("id"));
  }

  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
SO101SystemHardware::on_configure(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(rclcpp::get_logger("SO101SystemHardware"), "Configuring...");

  std::ifstream f(calib_file_);
  if (!f.is_open()) {
    RCLCPP_ERROR(
      rclcpp::get_logger("SO101SystemHardware"),
      "Calibration file not found: %s. Run: ros2 run so101_hardware "
      "calibrate_arm --arm follower --port %s",
      calib_file_.c_str(), port_.c_str());
    return hardware_interface::CallbackReturn::ERROR;
  }

  auto calib = nlohmann::json::parse(f);
  for (size_t i = 0; i < motor_ids_.size(); i++) {
    std::string id_str = std::to_string(motor_ids_[i]);
    homing_offsets_[motor_ids_[i]] = calib[id_str]["homing_offset"];
    range_mins_[motor_ids_[i]] = calib[id_str]["range_min"];
    range_maxes_[motor_ids_[i]] = calib[id_str]["range_max"];
  }

  RCLCPP_INFO(rclcpp::get_logger("SO101SystemHardware"), "Configured!");
  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
SO101SystemHardware::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;
  for (size_t i = 0; i < info_.joints.size(); i++) {
    state_interfaces.emplace_back(
      info_.joints[i].name,
      hardware_interface::HW_IF_POSITION,
      &hw_positions_[i]);
    state_interfaces.emplace_back(
      info_.joints[i].name,
      hardware_interface::HW_IF_VELOCITY,
      &hw_velocities_[i]);
  }
  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface>
SO101SystemHardware::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  for (size_t i = 0; i < info_.joints.size(); i++) {
    command_interfaces.emplace_back(
      info_.joints[i].name,
      hardware_interface::HW_IF_POSITION,
      &hw_commands_[i]);
  }
  return command_interfaces;
}

hardware_interface::CallbackReturn
SO101SystemHardware::on_activate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(rclcpp::get_logger("SO101SystemHardware"), "Activating...");

  if (!sms_sts_.begin(1000000, port_.c_str())) {
    RCLCPP_ERROR(
      rclcpp::get_logger("SO101SystemHardware"),
      "Failed to connect to motors on port %s", port_.c_str());
    return hardware_interface::CallbackReturn::ERROR;
  }
  bool cleanup_done = false;
  // Track EPROM unlock state per motor so the rollback can relock any motor
  // that was left unlocked when activation aborts (before the serial port is
  // closed). Insert on successful unlock, erase on successful lock.
  std::set<u8> unlocked_motors;
  const auto cleanup_activation = [this, &unlocked_motors, &cleanup_done]() {
      if (cleanup_done) {
        return;
      }
      const auto result = detail::rollback_activation(
        motor_ids_, unlocked_motors,
        [this](u8 id, u8 enabled) {
          return sms_sts_.EnableTorque(id, enabled);
        },
        [this](u8 id) {return sms_sts_.LockEprom(id);});
      if (!result.torque_disabled_all) {
        RCLCPP_ERROR(
          rclcpp::get_logger("SO101SystemHardware"),
          "Failed to disable torque for one or more motors during "
          "activation abort");
      }
      if (!result.eprom_relocked_all) {
        RCLCPP_ERROR(
          rclcpp::get_logger("SO101SystemHardware"),
          "Failed to relock EPROM for %zu motor(s) during activation "
          "abort; persistent parameters may be unprotected",
          result.relock_failures.size());
      }
      sms_sts_.end();
      cleanup_done = true;
    };
  detail::ActivationRollbackGuard rollback_guard(cleanup_activation);
  const auto abort_activation = [&cleanup_activation, &rollback_guard]() {
      cleanup_activation();
      rollback_guard.dismiss();
      return hardware_interface::CallbackReturn::ERROR;
    };

  // 0. Robustness check: Ping each motor to ensure it is connected and
  // responsive
  for (size_t i = 0; i < motor_ids_.size(); i++) {
    u8 id = motor_ids_[i];
    int retry = 3;
    bool found = false;
    while (retry--) {
      if (sms_sts_.Ping(id) != -1) {
        found = true;
        break;
      }
      usleep(10000); // 10ms wait between retries
    }

    if (!found) {
      RCLCPP_ERROR(
        rclcpp::get_logger("SO101SystemHardware"),
        "Motor ID %d is NOT responding! Check cables and power.",
        id);
      return abort_activation();
    }
    RCLCPP_DEBUG(
      rclcpp::get_logger("SO101SystemHardware"),
      "Motor ID %d found.", id);
  }

  const double TICKS_PER_RAD = 4096.0 / (2.0 * M_PI);

  // 1. Configure Hardware: Write Offsets, PID, and Return Delay
  const auto require_ack = [this](u8 id, const char * operation,
      const auto & write_operation) {
      for (int attempt = 1; attempt <= 3; ++attempt) {
        if (write_operation() != 0) {
          return true;
        }
        usleep(10000);
      }
      RCLCPP_ERROR(
        rclcpp::get_logger("SO101SystemHardware"),
        "Motor ID %d did not acknowledge %s; aborting activation", id,
        operation);
      return false;
    };
  for (size_t i = 0; i < motor_ids_.size(); i++) {
    u8 id = motor_ids_[i];

    // 1.1 Disable torque before configuration
    if (!require_ack(
        id, "disable torque",
        [this, id]() {return sms_sts_.EnableTorque(id, 0);}))
    {
      return abort_activation();
    }
    usleep(2000); // Small delay

    // 1.2 Unlock EPROM to allow parameter writing
    if (!require_ack(
        id, "unlock EPROM",
        [this, id]() {return sms_sts_.unLockEprom(id);}))
    {
      return abort_activation();
    }
    unlocked_motors.insert(id); // EPROM is now writable; must be relocked
    usleep(2000);

    // CORRECT Sign-Magnitude encoding for STS series (Sign bit is 11)
    int offset = homing_offsets_[id];
    u16 encoded_offset = (offset < 0) ?
      (static_cast<u16>(std::abs(offset)) | (1 << 11)) :
      static_cast<u16>(offset);

    RCLCPP_DEBUG(
      rclcpp::get_logger("SO101SystemHardware"),
      "Setting ID %d: Homing Offset=%d (Encoded: %u)", id, offset,
      encoded_offset);

    if (!require_ack(
        id, "write homing offset",
        [this, id, encoded_offset]() {
          return sms_sts_.writeWord(id, 31, encoded_offset);
        }) ||
      !require_ack(
        id, "write minimum range",
        [this, id]() {
          return sms_sts_.writeWord(id, 9, range_mins_[id]);
        }) ||
      !require_ack(
        id, "write maximum range",
        [this, id]() {
          return sms_sts_.writeWord(id, 11, range_maxes_[id]);
        }) ||
      !require_ack(
        id, "write response delay",
        [this, id]() {return sms_sts_.writeByte(id, 7, 0);}) ||
      !require_ack(
        id, "write position P gain",
        [this, id]() {return sms_sts_.writeByte(id, 21, 16);}) ||
      !require_ack(
        id, "write position D gain",
        [this, id]() {return sms_sts_.writeByte(id, 22, 32);}) ||
      !require_ack(
        id, "write position I gain",
        [this, id]() {return sms_sts_.writeByte(id, 23, 0);}))
    {
      return abort_activation();
    }
    usleep(2000);

    // 1.3 Lock EPROM after configuration to persist parameters
    if (!require_ack(
        id, "lock EPROM",
        [this, id]() {return sms_sts_.LockEprom(id);}))
    {
      return abort_activation();
    }
    unlocked_motors.erase(id); // EPROM relocked during normal flow
    usleep(2000);

    // 1.4 Enable torque after configuration
    if (!require_ack(
        id, "enable torque",
        [this, id]() {return sms_sts_.EnableTorque(id, 1);}))
    {
      return abort_activation();
    }
    usleep(2000);
  }

  // 2. Initialize sync read buffer for the extended position/speed/current
  // frame. NOTE: SMS_STS::syncReadBegin returns void (verified in SCS.h); it
  // only allocates the SDK receive buffer and stores the timeout, so it is not
  // a fail-closed gate. The gate is the Tx/Rx round below.
  sms_sts_.syncReadBegin(
    motor_ids_.size(), FEEDBACK_READ_LEN,
    FEEDBACK_READ_TIMEOUT_MS);
  current_node_ = rclcpp::Node::make_shared("so101_joint_current_publisher");
  current_pub_ =
    current_node_->create_publisher<ibrobot_msgs::msg::JointCurrent>(
    "/so101_follower/joint_currents", 10);

  // Initial sync: seed hw_commands_/positions_/velocities_/currents_ from real
  // feedback. Fail-closed: if the sync-read transmit fails or ANY motor fails
  // to return a full packet, on_activate must abort (torque off / EPROM relock
  // / port close) via abort_activation. We never dismiss the rollback guard
  // with uninitialised state.
  const auto sync_outcome = detail::perform_initial_sync_feedback(
    motor_ids_, has_reset_positions_, reset_positions_, TICKS_PER_RAD,
    CURRENT_RAW_TO_AMPERE,
    [this]() {
      return sms_sts_.syncReadPacketTx(
        motor_ids_.data(), motor_ids_.size(),
        FEEDBACK_START_ADDR, FEEDBACK_READ_LEN);
    },
    [this](u8 id, detail::FeedbackSample & out) -> bool {
      u8 data[FEEDBACK_READ_LEN];
      if (sms_sts_.syncReadPacketRx(id, data) != FEEDBACK_READ_LEN) {
        return false;
      }
      out = parse_feedback_packet(sms_sts_);
      return true;
    },
    hw_commands_, hw_positions_, hw_velocities_, hw_currents_);
  if (!sync_outcome.success) {
    if (sync_outcome.tx_failed) {
      RCLCPP_ERROR(
        rclcpp::get_logger("SO101SystemHardware"),
        "Initial sync read transmit failed; aborting activation");
    } else {
      RCLCPP_ERROR(
        rclcpp::get_logger("SO101SystemHardware"),
        "Initial sync read for motor ID %d failed; aborting activation",
        sync_outcome.failed_motor_id);
    }
    return abort_activation();
  }
  if (has_reset_positions_) {
    // Safety-relevant: with reset_positions configured the arm will move to
    // the configured safe pose rather than hold its current position.
    RCLCPP_INFO(
      rclcpp::get_logger("SO101SystemHardware"),
      "Initial commands set to configured reset_positions");
  }
  publish_currents(current_node_->get_clock()->now());

  RCLCPP_INFO(
    rclcpp::get_logger("SO101SystemHardware"),
    "Activated! Control loop running.");
  rollback_guard.dismiss();
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn
SO101SystemHardware::on_deactivate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(rclcpp::get_logger("SO101SystemHardware"), "Deactivating...");
  for (size_t i = 0; i < motor_ids_.size(); i++) {
    sms_sts_.EnableTorque(motor_ids_[i], 0);
  }
  usleep(100000);
  sms_sts_.syncReadEnd();
  sms_sts_.end();
  current_pub_.reset();
  current_node_.reset();
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type
SO101SystemHardware::read(const rclcpp::Time & time, const rclcpp::Duration &)
{
  static rclcpp::Clock steady_clock(RCL_STEADY_TIME);

  int read_len =
    sms_sts_.syncReadPacketTx(
    motor_ids_.data(), motor_ids_.size(),
    FEEDBACK_START_ADDR, FEEDBACK_READ_LEN);
  if (read_len <= 0) {
    RCLCPP_WARN_THROTTLE(
      rclcpp::get_logger("SO101SystemHardware"),
      steady_clock, 500, "SyncRead PacketTx FAILED");
    return hardware_interface::return_type::OK;
  }

  const double TICKS_PER_RAD = 4096.0 / (2.0 * M_PI);

  for (size_t i = 0; i < motor_ids_.size(); i++) {
    u8 data[FEEDBACK_READ_LEN];
    if (sms_sts_.syncReadPacketRx(motor_ids_[i], data) == FEEDBACK_READ_LEN) {
      const auto feedback = parse_feedback_packet(sms_sts_);

      hw_positions_[i] =
        (static_cast<double>(feedback.position) - 2048.0) / TICKS_PER_RAD;
      hw_velocities_[i] = static_cast<double>(feedback.speed) / TICKS_PER_RAD;
      hw_currents_[i] =
        static_cast<double>(feedback.current) * CURRENT_RAW_TO_AMPERE;
    }
  }
  publish_currents(time);
  return hardware_interface::return_type::OK;
}

void SO101SystemHardware::publish_currents(const rclcpp::Time & stamp)
{
  if (!current_pub_) {
    return;
  }

  ibrobot_msgs::msg::JointCurrent msg;
  msg.header.stamp = stamp;
  msg.name.reserve(info_.joints.size());
  msg.current.reserve(hw_currents_.size());
  for (size_t i = 0; i < info_.joints.size(); i++) {
    msg.name.push_back(info_.joints[i].name);
    msg.current.push_back(hw_currents_[i]);
  }
  current_pub_->publish(msg);
}

hardware_interface::return_type
SO101SystemHardware::write(const rclcpp::Time &, const rclcpp::Duration &)
{
  static rclcpp::Clock steady_clock(RCL_STEADY_TIME);
  const double TICKS_PER_RAD = 4096.0 / (2.0 * M_PI);

  for (size_t i = 0; i < motor_ids_.size(); i++) {
    double target_raw = hw_commands_[i] * TICKS_PER_RAD + 2048.0;

    // Safety clamp to [0, 4095]
    if (target_raw < 0) {
      target_raw = 0;
    }
    if (target_raw > 4095) {
      target_raw = 4095;
    }

    target_positions_[i] = static_cast<s16>(target_raw);
    target_speeds_[i] = 2400;
    target_accs_[i] = 50;
  }

  sms_sts_.SyncWritePosEx(
    motor_ids_.data(), motor_ids_.size(),
    target_positions_.data(), target_speeds_.data(),
    target_accs_.data());

  return hardware_interface::return_type::OK;
}

} // namespace so101_hardware

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
  so101_hardware::SO101SystemHardware,
  hardware_interface::SystemInterface)
