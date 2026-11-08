"""ROS client for a fenced pick/place handover; no robot-specific types."""

import time
import uuid

from .grasp_pipeline.contracts import GraspFailure
from .held_custody import CLAIM, RETURN, TOPIC, held_from_msg


class HeldObjectClient:
    def __init__(self, node):
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.qos import DurabilityPolicy, QoSProfile

        from ibrobot_msgs.msg import HeldObject
        from ibrobot_msgs.srv import ClaimHeldObject, ReturnHeldObject

        self.latest = None
        self.owner = f"{node.get_name()}/{uuid.uuid4().hex}"
        self.types = ClaimHeldObject, ReturnHeldObject
        self.group = ReentrantCallbackGroup()
        self.subscription = node.create_subscription(
            HeldObject,
            TOPIC,
            self._receive,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
            callback_group=self.group,
        )
        self.claim_client = node.create_client(ClaimHeldObject, CLAIM, callback_group=self.group)
        self.return_client = node.create_client(ReturnHeldObject, RETURN, callback_group=self.group)
        self.token, self.held = "", None

    def _receive(self, message):
        self.latest = message

    @staticmethod
    def _call(client, request, deadline, cancelled):
        if not client.wait_for_service(timeout_sec=max(0.0, min(1.0, deadline - time.monotonic()))):
            raise GraspFailure("CUSTODY_UNAVAILABLE", f"{client.srv_name} unavailable")
        future = client.call_async(request)
        while not future.done():
            if cancelled() or time.monotonic() >= deadline:
                # A timed-out claim may have reached pick. It expires without motion.
                raise GraspFailure("CANCELED" if cancelled() else "CUSTODY_TIMEOUT", "custody response not confirmed")
            time.sleep(0.002)
        response = future.result()
        if response is None or not response.success:
            raise GraspFailure(getattr(response, "reason_code", "CUSTODY_UNAVAILABLE"), "custody request refused")
        return response

    def claim(self, target, binding, *, deadline, cancelled):
        latest = self.latest
        if latest is None or not latest.holding:
            raise GraspFailure("NOT_HOLDING", "no picked object is available")
        request = self.types[0].Request(
            custody_id=latest.custody_id,
            target_query=target,
            claimant=self.owner,
            lease_s=min(120.0, max(1.0, deadline - time.monotonic() + 3.0)),
            dispatch_binding=binding,
        )
        response = self._call(self.claim_client, request, deadline, cancelled)
        self.token, self.held = response.claim_token, held_from_msg(response.held_object)
        return self.held

    def returned(self, outcome, reason=""):
        if self.held is None or not self.token:
            return
        request = self.types[1].Request(
            custody_id=self.held.custody_id, claim_token=self.token, outcome=outcome, reason_code=reason
        )
        self._call(self.return_client, request, time.monotonic() + 2.0, lambda: False)
        self.token, self.held = "", None
