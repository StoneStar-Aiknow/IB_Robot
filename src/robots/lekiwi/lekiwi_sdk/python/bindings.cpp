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

// Minimal pybind11 bindings for lekiwi_sdk: exposes OmniBase for the
// lekiwi_backend node. The shared bus stays in C++ (the backend uses the
// so101_sdk Arm bindings for the arm portion and constructs the OmniBase
// over the same bus).

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "feetech/bus.hpp"
#include "lekiwi/omni_base.hpp"

namespace py = pybind11;
using namespace lekiwi;

namespace
{

constexpr double kPi = 3.14159265358979323846;

BaseGeometry default_geometry()
{
  BaseGeometry g;
  g.wheel_radius = 0.05;
  g.base_radius = 0.125;
  g.max_wheel_radps = 4.602;
  g.mount_angles = {
    (240.0 * kPi / 180.0) - (kPi / 2.0),
    0.0 - (kPi / 2.0),
    (120.0 * kPi / 180.0) - (kPi / 2.0)};
  return g;
}

}  // namespace

PYBIND11_MODULE(lekiwi_sdk_py, m)
{
  m.doc() = "Minimal Python bindings for lekiwi_sdk (SimBus, OmniBase)";

  // --- SimBus: standalone simulated Feetech bus for tests/backends ---------
  py::class_<feetech::Bus, std::unique_ptr<feetech::Bus, py::nodelete>>(
    m, "SimBus", "Simulated Feetech bus (constructs with simulated=true)")
    .def(py::init([](std::vector<std::uint8_t> wheel_ids) {
        feetech::BusOptions options;
        options.simulated = true;
        for (const auto id : wheel_ids) {
          feetech::MotorConfig motor;
          motor.id = id;
          motor.mode = feetech::Mode::Wheel;
          options.motors.push_back(motor);
        }
        auto * bus = new feetech::Bus(options);
        bus->open();
        return bus;
    }), py::arg("wheel_ids"))
    .def("close", &feetech::Bus::close)
    .def("apply_configs", [](feetech::Bus & self) { return self.apply_all_configs(); })
    .def("sim", [](feetech::Bus & self) -> feetech::Bus::SimControl & {
        return self.sim();
    }, py::return_value_policy::reference_internal);

  // --- SimControl: NOT re-registered here (already registered by
  // so101_sdk_py). The lekiwi SimBus's sim() returns the same type.
  // py::class_<feetech::Bus::SimControl> must not appear twice across
  // modules in the same process.

  py::class_<BaseGeometry>(m, "BaseGeometry")
    .def(py::init(&default_geometry))
    .def_readwrite("wheel_radius", &BaseGeometry::wheel_radius)
    .def_readwrite("base_radius", &BaseGeometry::base_radius)
    .def_readwrite("max_wheel_radps", &BaseGeometry::max_wheel_radps)
    .def_readwrite("mount_angles", &BaseGeometry::mount_angles);

  // Pure kinematics (no bus) for runtime nodes that drive ros2_control
  // controllers instead of the SDK bus: body <-> wheel conversions and
  // odometry integration share one implementation with OmniBase.
  m.def(
    "body_to_wheel_velocities",
    [](double vx, double vy, double vtheta, const BaseGeometry & geometry) {
      const auto wheels = body_to_wheel_velocities(vx, vy, vtheta, geometry);
      return std::vector<double>(wheels.begin(), wheels.end());
    },
    py::arg("vx"), py::arg("vy"), py::arg("vtheta"), py::arg("geometry"));
  m.def(
    "wheel_deltas_to_body",
    [](const std::vector<double> & deltas, const BaseGeometry & geometry) -> py::object {
      if (deltas.size() != 3) {
        throw std::invalid_argument("wheel_deltas_to_body expects exactly 3 wheel values");
      }
      double dx = 0.0, dy = 0.0, dtheta = 0.0;
      if (!wheel_deltas_to_body({deltas[0], deltas[1], deltas[2]}, geometry, dx, dy, dtheta)) {
        return py::none();
      }
      return py::make_tuple(dx, dy, dtheta);
    },
    py::arg("wheel_deltas"), py::arg("geometry"));
  m.def(
    "integrate_pose",
    [](BasePose & pose, double dx, double dy, double dtheta) { integrate_pose(pose, dx, dy, dtheta); },
    py::arg("pose"), py::arg("dx"), py::arg("dy"), py::arg("dtheta"));

  py::class_<BasePose>(m, "BasePose")
    .def(py::init<>())
    .def_readwrite("x", &BasePose::x)
    .def_readwrite("y", &BasePose::y)
    .def_readwrite("theta", &BasePose::theta);

  py::class_<OmniBase>(m, "OmniBase")
    .def(py::init<feetech::Bus &, const std::vector<std::uint8_t> &, BaseGeometry>(),
         py::arg("bus"), py::arg("wheel_ids"), py::arg("geometry"))
    .def("activate", &OmniBase::activate)
    .def("set_wheel_velocities", [](OmniBase & self, double w0, double w1, double w2) {
        return self.set_wheel_velocities({w0, w1, w2});
    })
    .def("set_body_velocity", &OmniBase::set_body_velocity,
         py::arg("vx"), py::arg("vy"), py::arg("vtheta"))
    .def("read_wheels", [](OmniBase & self) -> py::object {
        std::array<double, 3> positions{}, velocities{};
        if (!self.read_wheels(positions, velocities)) {
            return py::none();
        }
        py::dict result;
        result["positions"] = py::make_tuple(positions[0], positions[1], positions[2]);
        result["velocities"] = py::make_tuple(velocities[0], velocities[1], velocities[2]);
        return result;
    })
    .def("update_odometry", [](OmniBase & self, BasePose & pose) -> bool {
        return self.update_odometry(pose);
    }, py::arg("pose"))
    .def("stop", &OmniBase::stop)
    .def("geometry", &OmniBase::geometry, py::return_value_policy::reference_internal);
}
