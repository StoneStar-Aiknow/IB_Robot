"""Bounded recovery transport, independent of ordinary execution capacity."""

from __future__ import annotations

import time
from contextlib import suppress
from uuid import uuid4

from .runtime_protocol import ControlError, encode_control, uuid_value

# Per resource public recovery transactions, not individual Host commands.
# ACKed/superseded receipts still occupy capacity through their retry window;
# unacknowledged Close and unresolved recovery retain their evidence.
MAX_RESOURCE_RECOVERY_RECEIPTS = 64
# Stream identifiers are opaque, but must leave room for all control envelopes.
MAX_RECOVERY_STREAM_BYTES = 1024


class ResourceRecovery:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.core = endpoint.core
        self.records = {}

    def settle_retired(self):
        """A proven retirement settles its owner's completed non-Close receipts."""
        for record in self.records.values():
            request = record["request"]
            if (
                record["state"] == "completed"
                and request["command"] != "close"
                and self.core.host_retired(request.get("target_id"), request.get("boot_id"))
            ):
                # Keep the immutable result through its existing replay window.
                # UNKNOWN work and other boots remain owned and quarantined.
                record["acknowledged"] = True

    def _discard_transport(self, record):
        """Remove every outstanding ROS future owned by a recovery record."""
        for target_id, command in record.get("commands", {}).items():
            future = command.get("future")
            if future is None or future.done():
                command["future"] = None
                continue
            client = self.endpoint._clients.get(target_id)
            remove = getattr(client, "remove_pending_request", None)
            if callable(remove):
                # Concurrent completion may already have removed the request.
                with suppress(Exception):
                    remove(future)
            command["future"] = None

    def request(self, message):
        """Admit recoveries only if all envelopes fit; stream IDs are at most 1024 UTF-8 bytes."""
        message = dict(message)
        resume_until = message.pop("resume_until_ns", None)
        acknowledged = message.pop("acknowledged", False)
        if type(acknowledged) is not bool:
            raise ValueError("recovery acknowledgement must be boolean")
        resource, operation_id = message["resource"], message["operation_id"]
        uuid_value(operation_id)
        expiry = message["accept_until_ns"]
        now = self.core.now_ns()
        previous = self.records.get(operation_id)
        if previous is not None:
            if previous["request"] != message:
                raise ControlError("OPERATION_CONFLICT")
            if acknowledged:
                if previous["state"] not in {"completed", "superseded"}:
                    raise ControlError("RECOVERY_EVIDENCE_PENDING")
                previous["acknowledged"] = True
                return {"state": "acknowledged"}
            if previous["state"] == "superseded":
                # A completed Close remains complete even if another owner
                # finished the resource activation that its waiter missed.
                closed = previous.get("closed_targets", ())
                if (
                    message["command"] == "close"
                    and message.get("target_id") in closed
                    and all(self.core.host_retired(key, previous["identities"][key]) for key in closed)
                ):
                    return {"state": "completed", "phase": "close", "error": ""}
                raise ControlError("RECOVERY_SUPERSEDED")
            if resume_until is not None:
                caller = self.core.targets.get(message.get("target_id"))
                window = (
                    caller.retry_window_ms if caller else min(target.retry_window_ms for target in previous["targets"])
                ) * 1000000
                if type(resume_until) is not int or not now < resume_until <= now + window:
                    raise ControlError("OPERATION_EXPIRED")
                if previous["state"] != "completed" and resume_until > previous.get("resume_until_ns", 0):
                    if any(
                        key != operation_id
                        and record["state"] == "running"
                        and record["request"]["resource"] == resource
                        for key, record in self.records.items()
                    ):
                        raise ControlError("RECOVERY_BUSY")
                    if any(
                        self.core.host_boot(target_id) != boot_id
                        for target_id, boot_id in previous["identities"].items()
                    ):
                        raise ControlError("RECOVERY_IDENTITY_MISMATCH")
                    # Continue the same phase and command IDs. In particular a
                    # lost Reset/Close reply must never execute that call twice.
                    # A reply still in flight belongs to the old acceptance
                    # window and may report OPERATION_EXPIRED. Re-query the
                    # same command IDs under the renewed window instead.
                    self._discard_transport(previous)
                    previous.update(
                        accept_until_ns=min(
                            resume_until, now + min(target.retry_window_ms for target in previous["targets"]) * 1000000
                        ),
                        state="running",
                        resume_until_ns=resume_until,
                    )
                    previous.pop("error", None)
                    self.core.fence_resource(resource)
            elif type(expiry) is not int or now >= expiry:
                raise ControlError("OPERATION_EXPIRED")
            # Membership may change after completion. A retained immutable
            # receipt can be replayed without repeating lifecycle calls.
            return {"state": previous["state"], "phase": previous["phase"], "error": previous.get("error", "")}
        if acknowledged:
            return {"state": "acknowledged"}
        if resume_until is not None:
            # An absent receipt cannot prove whether a lifecycle call ran.
            raise ControlError("RECOVERY_RECEIPT_UNAVAILABLE")
        targets = self.core.recovery_targets(resource)
        if not targets or message["command"] not in {"drain", "reset", "close", "reset_stream", "close_stream"}:
            raise ControlError("INVALID_RECOVERY_REQUEST")
        stream_control = message["command"] in {"reset_stream", "close_stream"}
        if stream_control and (
            not isinstance(message.get("stream_id"), str) or not message["stream_id"] or "target_id" not in message
        ):
            raise ControlError("INVALID_RECOVERY_STREAM")
        if stream_control and len(message["stream_id"].encode()) > MAX_RECOVERY_STREAM_BYTES:
            raise ControlError("INVALID_RECOVERY_STREAM")
        if not stream_control and "stream_id" in message:
            raise ControlError("INVALID_RECOVERY_STREAM")
        caller = self.core.targets.get(message.get("target_id"))
        retry_window_ms = caller.retry_window_ms if caller else min(t.retry_window_ms for t in targets)
        if type(expiry) is not int or not now < expiry <= now + retry_window_ms * 1000000:
            raise ControlError("OPERATION_EXPIRED")
        lifecycle_group = None
        if "target_id" in message:
            target = self.core.targets.get(message["target_id"])
            observed_boot = self.core.host_boot(message["target_id"])
            if (
                target is None
                or target not in targets
                or target.hardware_resource_id != resource
                or observed_boot is None
                or observed_boot != message.get("boot_id")
            ):
                raise ControlError("RECOVERY_IDENTITY_MISMATCH")
            lifecycle_group = target.lifecycle_group
        self.collect()
        # A quarantined device retains its evidence without consuming another
        # independent resource's recovery capacity. Membership is already bounded.
        # Reserve the last transaction for irreversible Close, which can retire
        # this owner even when failed stream mutations fill the replay ledger.
        if sum(
            r["request"]["resource"] == resource for r in self.records.values()
        ) >= MAX_RESOURCE_RECOVERY_RECEIPTS - (message["command"] != "close") or any(
            r["state"] == "running" and r["request"]["resource"] == resource for r in self.records.values()
        ):
            raise ControlError("RECOVERY_BUSY")
        # Validate complete ACK/renewal envelopes before fencing or cancelling
        # any work. The same Host builder is used below for actual sends.
        encode_control(
            {
                "verb": "recover",
                "epoch": self.core.epoch,
                **message,
                "acknowledged": True,
                "resume_until_ns": expiry + retry_window_ms * 1000000,
            }
        )
        for target in targets:
            record = {
                "phase": message["command"],
                "request": message,
                "identities": {target.target_id: self.core.host_boot(target.target_id)},
                "accept_until_ns": expiry + retry_window_ms * 1000000,
            }
            command = {"id": operation_id, "accept_until_ns": expiry, "result": {}}
            encode_control(self._command_payload(record, target, command))
        identities = self.core.begin_recovery(resource, targets)
        self.records[operation_id] = {
            "request": dict(message),
            # The caller knows its own retry window, while drain visits every
            # peer. Bound each transport window by every participant's limit.
            "accept_until_ns": min(expiry, now + min(t.retry_window_ms for t in targets) * 1000000),
            "state": "running",
            "phase": "drain",
            "targets": targets,
            "identities": identities,
            "lifecycle_group": lifecycle_group,
            "commands": {},
            "done": set(),
        }
        return {"state": "running", "phase": "drain"}

    def collect(self):
        """Expire only settled receipts, also when no new recovery arrives."""
        now = self.core.now_ns()
        retained = {}
        for key, record in self.records.items():
            # Superseded non-Close requests can no longer resume a mutation.
            # Close may still replay positive retirement evidence to its owner.
            settled = record.get("acknowledged", False) or (
                record["state"] == "superseded" and record["request"]["command"] != "close"
            )
            if not settled or record["accept_until_ns"] > now:
                retained[key] = record
            else:
                self._discard_transport(record)
        self.records = retained

    def _command_payload(self, record, target, command):
        phase = record["phase"]
        key = target.target_id
        return {
            "verb": "recover",
            "epoch": self.core.epoch,
            "operation_id": command["id"],
            "command": phase,
            **({"acknowledged": True} if "result" in command else {}),
            "boot_id": record["identities"][key],
            "lifecycle_group": target.lifecycle_group,
            "accept_until_ns": command["accept_until_ns"],
            **(
                {"resume_until_ns": record["accept_until_ns"]}
                if command["accept_until_ns"] != record["accept_until_ns"]
                else {}
            ),
            **({"expected_generation": command["generation"]} if phase == "activate" else {}),
            **({"stream_id": record["request"]["stream_id"]} if phase in {"reset_stream", "close_stream"} else {}),
        }

    def tick(self):
        from .runtime_ros import _RemoteControlError, _reply, _request

        self.collect()
        for record in self.records.values():
            if record["state"] != "running":
                continue
            if self.core.now_ns() >= record["accept_until_ns"]:
                self._discard_transport(record)
                record.update(state="unknown", error="RECOVERY_EXPIRED")
                continue
            if any(self.core.host_boot(key) != boot for key, boot in record["identities"].items()):
                self._discard_transport(record)
                record.update(state="unknown", error="RECOVERY_IDENTITY_MISMATCH")
                continue
            # A captured peer can finish ordinary Close while this resource
            # drains. Its boot-bound Global retirement is sufficient evidence;
            # never require its destroyed endpoint or reactivate that owner.
            targets = [
                target
                for target in record["targets"]
                if not self.core.host_retired(target.target_id, record["identities"][target.target_id])
            ]
            phase = record["phase"]
            if phase == "activate" and record["request"]["command"] == "close":
                # The closed lifecycle owner is never activated again. Only
                # peers drained for its resource barrier resume admission.
                targets = [target for target in targets if target.target_id not in record.get("closed_targets", ())]
            if phase == "evidence":
                if not self.core.resource_settled(record["request"]["resource"]):
                    continue
                command = record["request"]["command"]
                record["phase"] = "activate" if command == "drain" else command
                record["done"] = set()
                record["commands"] = {}
                continue
            if phase in {"reset_stream", "close_stream"}:
                targets = [target for target in targets if target.target_id == record["request"]["target_id"]]
            elif phase in {"reset", "close"}:
                owners = {}
                for target in targets:
                    if record["lifecycle_group"] is not None and target.lifecycle_group != record["lifecycle_group"]:
                        continue
                    owners.setdefault(target.lifecycle_group, target)
                targets = list(owners.values())
            for target in targets:
                key = target.target_id
                if key in record["done"]:
                    continue
                try:
                    client = self.endpoint._clients.get(key)
                    if client is None:
                        continue  # The spin owner must collect retired transport first.
                    command = record["commands"].get(key)
                    if command is None:
                        command = {
                            "id": str(uuid4()),
                            "future": None,
                            "sent": 0,
                            "accept_until_ns": record["accept_until_ns"],
                        }
                        record["commands"][key] = command
                    future = command["future"]
                    if future is not None:
                        if not future.done():
                            if time.monotonic_ns() - command["sent"] > self.endpoint.control_rpc_timeout_ns:
                                client.remove_pending_request(future)
                                command["future"] = None
                            continue
                        command["future"] = None
                        try:
                            response = _reply(future.result())
                            if response["state"] == "completed":
                                command["result"] = response
                            if response["state"] == "acknowledged" and "result" in command:
                                response = command["result"]
                                if response["state"] == "unknown":
                                    # Global now owns the failure evidence. Host
                                    # may collect its receipt after the replay
                                    # window; the resource remains quarantined.
                                    command.pop("result")
                                    self._discard_transport(record)
                                    record.update(state="unknown", error=response["error"])
                                    break
                                if phase == "activate":
                                    self.core.complete_activation(
                                        key,
                                        record["identities"][key],
                                        response["generation"],
                                        command["generation"],
                                        ready=response.get("ready", False),
                                    )
                                record["done"].add(key)
                                continue
                        except Exception as exc:
                            if self.core.host_retired(key, record["identities"][key]):
                                continue
                            if (
                                isinstance(exc, _RemoteControlError)
                                and exc.code == "RECOVERY_RECEIPT_UNAVAILABLE"
                                and phase in {"drain", "close", "activate"}
                            ):
                                # Only repeatable barriers may start a new
                                # command when history is unavailable. Close
                                # retains owner progress; activation reuses the
                                # coordinated generation. Never replay Reset.
                                del record["commands"][key]
                                continue
                            if (
                                phase == "activate"
                                and isinstance(exc, _RemoteControlError)
                                and exc.code == "RECOVERY_EVIDENCE_PENDING"
                            ):
                                # Global can observe terminal evidence before the
                                # host consumes its ACK. Retry the same activation
                                # on the next tick, within the original window.
                                continue
                            if (
                                isinstance(exc, _RemoteControlError)
                                and exc.code in {"RECOVERY_CAPACITY", "RECOVERY_WORKER_SUBMIT_FAILED"}
                                and exc.operation_started is False
                            ):
                                # There is no Host receipt to resume. Discard
                                # only this proven-unstarted command; keep the
                                # completed phases and the resource barrier.
                                del record["commands"][key]
                                continue
                            if (
                                phase in {"reset", "reset_stream"}
                                and isinstance(exc, _RemoteControlError)
                                and exc.operation_started is False
                            ):
                                # Retain the same receipt and identity. Host
                                # may continue only with explicit non-start
                                # evidence; UNKNOWN never enters this branch.
                                continue
                            if isinstance(exc, _RemoteControlError):
                                # Receipt ACK transfers evidence, not execution
                                # success. Reuse the successful-result handshake
                                # so failed mutations cannot pin Host capacity
                                # forever while Global retains UNKNOWN.
                                command["result"] = {"state": "unknown", "error": exc.code}
                                continue
                            self._discard_transport(record)
                            record.update(state="unknown", error=getattr(exc, "code", "RECOVERY_UNKNOWN"))
                            break
                    if client.service_is_ready():
                        if command["sent"] == 0:
                            # Discovery has never allowed a send. There cannot
                            # be a Host receipt to resume under the old window.
                            command["accept_until_ns"] = record["accept_until_ns"]
                        if phase == "activate":
                            expected = self.core.next_activation_generation(key, record["identities"][key])
                            command.setdefault("generation", expected)
                        request = _request(self._command_payload(record, target, command))
                        # A send exception is uncertain, unlike failed discovery.
                        command["sent"] = time.monotonic_ns()
                        command["future"] = client.call_async(request)
                except Exception as exc:
                    self._discard_transport(record)
                    record.update(state="unknown", error=getattr(exc, "code", "RECOVERY_TRANSPORT_FAILED"))
                    break
            if record["state"] != "running" or any(target.target_id not in record["done"] for target in targets):
                continue
            # A peer retired during this phase may still have a lost reply in
            # transport. Release that future before replacing the command map.
            self._discard_transport(record)
            if phase == "drain":
                record.update(phase="evidence", commands={}, done=set())
            elif phase in {"reset", "reset_stream", "close_stream"}:
                record.update(phase="activate", commands={}, done=set())
            elif phase == "close":
                # All resource operations crossed the evidence/ACK barrier
                # before Close. Its successful owner response now proves
                # cleanup for every stage in that lifecycle group.
                closed_targets = [
                    target
                    for target in record["targets"]
                    if record["lifecycle_group"] is None or target.lifecycle_group == record["lifecycle_group"]
                ]
                try:
                    self.core.retire_recovered(closed_targets, record["identities"])
                    self.settle_retired()
                except ControlError as exc:
                    record.update(state="unknown", error=exc.code)
                    continue
                record.update(
                    phase="activate",
                    commands={},
                    done=set(),
                    closed_targets={target.target_id for target in closed_targets},
                )
            else:
                record["state"] = "completed"
                if phase == "activate":
                    self.core.finish_recovery(record["request"]["resource"])
                    for previous in self.records.values():
                        if (
                            previous["state"] == "unknown"
                            and previous["request"]["resource"] == record["request"]["resource"]
                        ):
                            self._discard_transport(previous)
                            previous.update(state="superseded", error="RECOVERY_SUPERSEDED")
