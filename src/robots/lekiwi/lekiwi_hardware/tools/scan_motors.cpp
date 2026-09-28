// Copyright 2026 IB_Robot Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * Motor Scanner for LeKiwi
 * Scans all expected motor IDs (1-9) and reports which ones are responding.
 *
 * Rewritten on the shared feetech_sdk Bus: the previous implementation talked
 * to the vendored FTServo SDK directly (SMS_STS::Ping), which disappeared with
 * the unpinned FetchContent block. feetech_sdk exposes no Ping, so presence is
 * probed with a single-motor synchronized read -- a motor that answers a group
 * read of just itself is on the bus.
 */

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "feetech/bus.hpp"
#include "feetech/types.hpp"

namespace
{

constexpr int kFirstWheelId = 7;
constexpr int kLastMotorId = 9;
constexpr int kRetries = 3;

std::string describe(std::uint8_t id)
{
  if (id < kFirstWheelId) {
    return " (Arm joint " + std::to_string(id) + ")";
  }
  return " (Base wheel " + std::to_string(id - (kFirstWheelId - 1)) + ")";
}

}  // namespace

int main(int argc, char ** argv)
{
  std::string port = "/dev/ttyACM0";
  if (argc >= 2) {
    port = argv[1];
  }

  feetech::BusOptions options;
  options.port = port;
  for (int id = 1; id <= kLastMotorId; ++id) {
    feetech::MotorConfig motor;
    motor.id = static_cast<std::uint8_t>(id);
    motor.name = std::to_string(id);
    motor.mode = (id >= kFirstWheelId) ? feetech::Mode::Wheel : feetech::Mode::Position;
    options.motors.push_back(motor);
  }

  std::cout << "=== LeKiwi Motor Scanner ===" << std::endl;
  std::cout << "Scanning port: " << port << std::endl;
  std::cout << "Baud rate: " << options.baudrate << std::endl;
  std::cout << std::endl;

  feetech::Bus bus(options);
  if (!bus.open()) {
    std::cout << "ERROR: Failed to open the motor bus on " << port << std::endl;
    std::cout << "Please check:" << std::endl;
    std::cout << "  1. Is the device connected?" << std::endl;
    std::cout << "  2. Do you have permission to access " << port << "?" << std::endl;
    std::cout << "     Try: sudo chmod 666 " << port << std::endl;
    return 1;
  }

  std::cout << "Connected successfully. Scanning for motors..." << std::endl;
  std::cout << std::endl;
  std::cout << "Scanning expected motor IDs (1-" << kLastMotorId << "):" << std::endl;
  std::cout << "------------------------------------" << std::endl;

  // Probing one motor at a time keeps an absent motor from failing the scan
  // for the others (sync_read over a subset only fails on that subset).
  int found_count = 0;
  for (int id = 1; id <= kLastMotorId; ++id) {
    const std::vector<std::uint8_t> ids{static_cast<std::uint8_t>(id)};
    std::vector<feetech::MotorSample> samples;
    bool found = false;

    for (int retry = 0; retry < kRetries && !found; ++retry) {
      const feetech::MotorOpResult result = bus.sync_read(samples, ids);
      found = result.ok && !samples.empty() && samples.front().valid;
    }

    const std::uint8_t motor_id = static_cast<std::uint8_t>(id);
    if (found) {
      ++found_count;
      std::cout << "[OK ] Motor ID " << id << describe(motor_id) << std::endl;
    } else {
      std::cout << "[ X ] Motor ID " << id << " - NOT RESPONDING" << describe(motor_id) <<
        std::endl;
    }
  }

  std::cout << "------------------------------------" << std::endl;
  std::cout << "Found " << found_count << "/" << kLastMotorId << " motors" << std::endl;
  std::cout << std::endl;

  if (found_count < kLastMotorId) {
    std::cout << "WARNING: Some motors are not responding!" << std::endl;
    std::cout << std::endl;
    std::cout << "Possible solutions:" << std::endl;
    std::cout << "  1. Check physical connections" << std::endl;
    std::cout << "  2. Check if motors are powered" << std::endl;
    std::cout << "  3. Verify motor IDs match configuration" << std::endl;
    std::cout << "  4. Update the URDF motor ID configuration if needed" << std::endl;
    std::cout << "     File: src/robots/lekiwi/lekiwi_description/urdf/" << std::endl;
  } else {
    std::cout << "All motors are responding correctly!" << std::endl;
  }

  bus.close();
  return (found_count == kLastMotorId) ? 0 : 1;
}
