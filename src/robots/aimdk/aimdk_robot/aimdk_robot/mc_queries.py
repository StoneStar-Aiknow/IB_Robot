"""Read-only, typed queries of the MC; never fabricate success from cached state."""

from __future__ import annotations

import math
import threading

from aimdk_msgs.srv import GetMcAction, GetMcLocomotionSpeedStatus

from aimdk_robot import vendor_gateway as vg
from ibrobot_msgs.msg import LocomotionProfile
from ibrobot_msgs.srv import GetLocomotionProfiles, GetMotionState


def speed_profile(status) -> LocomotionProfile:
    result = LocomotionProfile(mode=int(status.speed_mode.value))
    for axis in ("forward", "lateral", "angular"):
        bounds = getattr(status, f"{axis}_bounds")
        values = [float(bounds.min_value), float(bounds.max_value)]
        if not all(math.isfinite(value) for value in values) or values[0] > values[1]:
            raise ValueError(f"invalid {axis} speed bounds: {values}")
        setattr(result, axis, values)
    return result


class _AsyncServiceCaller(vg.ServiceCaller):
    """Yield executor workers while retaining the vendor retry/readiness contract."""

    async def call(self, request):
        for _ in range(self._attempts):
            future = self._client.call_async(request)
            # rclpy callbacks have no asyncio loop; a wall timer also expires
            # when ROS time is paused. Cancellation resumes the awaiting task.
            timer = threading.Timer(self._timeout_s, future.cancel)
            timer.daemon = True
            timer.start()
            try:
                response = await future
                if future.cancelled() or response is None:
                    continue
                code = vg.response_code(response)
                return vg.VendorCallResult(code == 0, code, vg.response_message(response), response)
            except Exception as error:
                return vg.VendorCallResult(False, -1, f"MC query failed: {error}")
            finally:
                timer.cancel()
                future.cancel()
                self._client.remove_pending_request(future)
        return vg.VendorCallResult(False, -1, "vendor service did not answer")


class McQueries:
    """Optional public query endpoints, owned and declared by the runtime profile."""

    def __init__(self, node, interfaces) -> None:
        self._node = node
        services = node.vendor_config.get("services") or {}
        for key, interface, vendor_type, public_type, callback in (
            ("get_action", "motion.state", GetMcAction, GetMotionState, self._action),
            (
                "get_speed_profiles",
                "motion.locomotion_profiles",
                GetMcLocomotionSpeedStatus,
                GetLocomotionProfiles,
                self._speeds,
            ),
        ):
            if interface not in interfaces:
                continue
            client = node.create_client(vendor_type, services[key], callback_group=node.action_callback_group)
            caller = _AsyncServiceCaller(node, client, attempts=2, timeout_s=0.5)
            setattr(self, f"_{key}", (vendor_type, caller))
            node.create_service(
                public_type, str(interfaces[interface]["endpoint"]), callback, callback_group=node.action_callback_group
            )

    async def _query(self, key):
        vendor_type, caller = getattr(self, f"_{key}")
        if not caller.ready():
            return vg.VendorCallResult(False, -1, "MC query service unavailable")
        request = vendor_type.Request()
        request.request.header.stamp = self._node.get_clock().now().to_msg()
        return await caller.call(request)

    async def _action(self, _request, response):
        outcome = await self._query("get_action")
        response.success, response.message = outcome.ok, outcome.message
        if outcome.ok:
            info = outcome.response.info
            response.action_name = info.action_desc
            response.action_id = int(info.current_action.value)
            response.action_status = int(info.status.value)
        return response

    async def _speeds(self, _request, response):
        outcome = await self._query("get_speed_profiles")
        response.success, response.message = outcome.ok, outcome.message
        if not outcome.ok:
            return response
        names = ["none", "low", "medium", "high", "step"]
        try:
            current = speed_profile(outcome.response.current_speed_status)
            profiles = [speed_profile(getattr(outcome.response, f"{name}_speed_status")) for name in names]
        except ValueError as error:
            response.success, response.message = False, str(error)
            return response
        response.current, response.names, response.profiles = current, names, profiles
        return response
