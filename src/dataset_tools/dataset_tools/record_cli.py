#!/usr/bin/env python3

"""
Command Line Interface for triggering episodic recordings.
Sends Action goals to the EpisodeRecorderServer.
"""

import json
import sys
import threading
import traceback

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Empty, Trigger

# Import the global interface
from ibrobot_msgs.action import RecordEpisode


class RecordCLI(Node):
    def __init__(self):
        super().__init__("record_cli")

        self.declare_parameter("dispatcher_reset_service", "/action_dispatcher/reset")
        self.declare_parameter("policy_reset_service", "/inference/policy/reset")
        self.declare_parameter("reset_timeout_sec", 2.0)
        self.declare_parameter("control_mode", "teleop")
        self.declare_parameter("reset_before_episode", "auto")
        self._reset_timeout_s = float(self.get_parameter("reset_timeout_sec").value)
        self._episode_finished_evt = threading.Event()
        self._goal_started_evt = threading.Event()
        self._goal_rejected_evt = threading.Event()
        self._last_result_success = False
        self._last_result_message = ""

        # Action client to start recording
        self._action_client = ActionClient(self, RecordEpisode, "record_episode")

        # Service client to stop recording early
        self._cancel_client = self.create_client(Trigger, "record_episode/cancel")
        self._dispatcher_reset_client = self.create_client(
            Empty,
            self.get_parameter("dispatcher_reset_service").value,
        )
        self._policy_reset_client = self.create_client(
            Trigger,
            self.get_parameter("policy_reset_service").value,
        )
        # Service client to get dataset info
        self._info_client = self.create_client(Trigger, "record_episode/get_info")
        self._last_episode_client = self.create_client(Trigger, "record_episode/get_last_episode")
        self._delete_last_client = self.create_client(Trigger, "record_episode/delete_last")

        self.get_logger().info("Record CLI started. Waiting for Action Server...")
        self._action_client.wait_for_server()
        self.get_logger().info("Connected to Episode Recorder Server!")

    def send_goal(self, prompt_text: str):
        self._goal_started_evt.clear()
        self._goal_rejected_evt.clear()
        self._last_result_success = False
        self._last_result_message = ""
        if self._should_reset_before_episode():
            self.prepare_new_episode()

        goal_msg = RecordEpisode.Goal()
        goal_msg.prompt = prompt_text

        self.get_logger().info(f"Sending goal with prompt: '{prompt_text}'")

        # We don't block here so the user can cancel it
        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning("Goal rejected by server (Is it already recording?)")
            self._goal_rejected_evt.set()
            self._episode_finished_evt.set()
            return

        self._goal_started_evt.set()
        self.get_logger().info("🔴 RECORDING STARTED.")
        print("Controls while recording:")
        print("  Enter       stop and review")
        print("  d + Enter   stop, discard, then return to Prompt")
        print("  r + Enter   stop, discard, then retry same prompt")

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        try:
            result_wrapper = future.result()
            result = result_wrapper.result
            self._last_result_success = bool(result.success)
            self._last_result_message = str(result.message)
            if result.success:
                self.get_logger().info(f"✅ RECORDING FINALIZED: {result.message}")
            else:
                self.get_logger().info(f"⚠️  RECORDING CANCELLED/ENDED: {result.message}")
        except Exception as e:
            self.get_logger().error(f"Action failed to get result: {e}")

        print("\n----------------------------------------")
        print("Episode finalized! Ready for next episode.")
        self._episode_finished_evt.set()

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        # Optional: Print progress on the same line
        sys.stdout.write(f"\r[Time Left: {feedback.seconds_remaining}s] {feedback.feedback_message}   ")
        sys.stdout.flush()

    def cancel_recording(self):
        if not self._cancel_client.service_is_ready():
            self.get_logger().error("Cancel service not available right now.")
            return

        req = Trigger.Request()
        future = self._cancel_client.call_async(req)
        future.add_done_callback(self._cancel_response_callback)

    def _cancel_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("Stop signal acknowledged.")
            else:
                self.get_logger().error(f"Stop signal failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Cancel service call failed: {e}")

    def prepare_new_episode(self):
        """Best-effort reset so a new episode reuses clean policy/dispatcher state."""
        # Prefer dispatcher reset because it clears queued action chunks and then
        # asks inference_service to clear policy-local caches. The direct policy
        # reset remains a fallback for deployments that start record_cli without
        # action_dispatch.
        if self._reset_dispatcher_state():
            return
        self._reset_policy_state()

    def _should_reset_before_episode(self) -> bool:
        override = str(self.get_parameter("reset_before_episode").value).strip().lower()
        if override in {"true", "1", "yes", "on"}:
            return True
        if override in {"false", "0", "no", "off"}:
            return False
        return str(self.get_parameter("control_mode").value).strip() == "model_inference"

    def _reset_dispatcher_state(self) -> bool:
        service_name = self.get_parameter("dispatcher_reset_service").value
        if not service_name:
            return False
        if not self._dispatcher_reset_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warning(f"Dispatcher reset service unavailable: {service_name}")
            return False

        try:
            future = self._dispatcher_reset_client.call_async(Empty.Request())
        except Exception as e:
            self.get_logger().warning(f"Dispatcher reset call failed: {e}")
            return False

        if not self._wait_for_future(future, timeout_sec=self._reset_timeout_s):
            self.get_logger().warning("Dispatcher reset call timed out.")
            return False

        try:
            response = future.result()
        except Exception as e:
            self.get_logger().warning(f"Dispatcher reset call failed: {e}")
            return False

        if response is not None:
            self.get_logger().info("Dispatcher state reset for new episode.")
            return True

        self.get_logger().warning("Dispatcher reset call failed.")
        return False

    def _reset_policy_state(self) -> bool:
        service_name = self.get_parameter("policy_reset_service").value
        if not service_name:
            return False
        if not self._policy_reset_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warning(f"Policy reset service unavailable: {service_name}")
            return False

        try:
            future = self._policy_reset_client.call_async(Trigger.Request())
        except Exception as e:
            self.get_logger().warning(f"Policy reset call failed: {e}")
            return False

        if not self._wait_for_future(future, timeout_sec=self._reset_timeout_s):
            self.get_logger().warning("Policy reset call timed out.")
            return False

        try:
            response = future.result()
        except Exception as e:
            self.get_logger().warning(f"Policy reset call failed: {e}")
            return False

        if response is not None and response.success:
            self.get_logger().info("Policy runtime state reset for new episode.")
            return True

        message = response.message if response is not None else ""
        self.get_logger().warning(f"Policy reset call failed. {message}")
        return False

    def delete_last_episode(self, timeout_sec: float = 5.0) -> bool:
        """Ask the recorder server to delete the last finalized episode."""
        if not self._delete_last_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning("Delete service unavailable: record_episode/delete_last")
            return False
        try:
            future = self._delete_last_client.call_async(Trigger.Request())
        except Exception as e:
            self.get_logger().warning(f"Delete service call failed: {e}")
            return False
        if not self._wait_for_future(future, timeout_sec=timeout_sec):
            self.get_logger().warning("Delete service call timed out.")
            return False
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().warning(f"Delete service call failed: {e}")
            return False
        if response is None:
            self.get_logger().warning("Delete service returned no response.")
            return False
        if response.success:
            info = _parse_json(response.message) or {}
            episode_dir = info.get("episode_dir", response.message)
            dataset_root = info.get("dataset_root", "")
            print(f"🗑️  DISCARDED {episode_dir}")
            if dataset_root:
                print(f"📁 Dataset remains: {dataset_root}")
            return True
        self.get_logger().warning(f"Delete service refused: {response.message}")
        return False

    def get_last_episode_info(self, timeout_sec: float = 3.0) -> dict | None:
        """Return the last finalized episode info from the recorder server."""
        if not self._last_episode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning("Last episode service unavailable: record_episode/get_last_episode")
            return None
        try:
            future = self._last_episode_client.call_async(Trigger.Request())
        except Exception as e:
            self.get_logger().warning(f"Last episode service call failed: {e}")
            return None
        if not self._wait_for_future(future, timeout_sec=timeout_sec):
            self.get_logger().warning("Last episode service call timed out.")
            return None
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().warning(f"Last episode service call failed: {e}")
            return None
        if response is None or not response.success:
            message = response.message if response is not None else "no response"
            self.get_logger().warning(f"Last episode unavailable: {message}")
            return None
        info = _parse_json(response.message)
        if info is None:
            self.get_logger().warning(f"Last episode response was not valid JSON: {response.message}")
            return None
        return info

    @staticmethod
    def _wait_for_future(future, timeout_sec: float) -> bool:
        """Wait for a future while the node is already spun by the executor thread."""
        if future.done():
            return True
        done_event = threading.Event()
        future.add_done_callback(lambda _future: done_event.set())
        if future.done():
            return True
        return done_event.wait(timeout=timeout_sec)


def _parse_json(value: str) -> dict | None:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _recording_command(raw: str) -> str:
    cmd = raw.strip().lower()
    if cmd in {"", "q", "quit"}:
        return "review"
    if cmd in {"d", "discard"}:
        return "discard"
    if cmd in {"r", "retry"}:
        return "retry"
    print(f"⚠️  Unknown recording command '{raw}'. Stopping for review.")
    return "review"


def _review_choice(raw: str) -> str | None:
    cmd = raw.strip().lower()
    if cmd in {"", "s", "save"}:
        return "save"
    if cmd in {"d", "discard"}:
        return "discard"
    if cmd in {"r", "retry"}:
        return "retry"
    if cmd in {"q", "quit"}:
        return "quit_keep"
    return None


def _episode_label(info: dict | None) -> str:
    if not info:
        return "episode"
    return str(info.get("episode_name") or info.get("episode_dir") or "episode")


def _print_episode_review(info: dict | None, prompt: str) -> None:
    print("\n========================================")
    print("Episode stopped and finalized.")
    if info:
        print(f"Dataset: {info.get('dataset_root', 'Unknown')}")
        print(f"Episode: {info.get('episode_dir', 'Unknown')}")
        print(f"Messages written: {info.get('messages', 'Unknown')}")
        print(f"Prompt: {info.get('prompt') or prompt}")
    else:
        print("⚠️  Episode path unavailable from recorder server.")
        print(f"Prompt: {prompt}")
    print("")
    print("[s] save / [d] discard / [r] discard and retry / [q] quit and keep")


def _print_kept_episode(info: dict | None) -> None:
    print(f"✅ KEPT {_episode_label(info)}")
    if info and info.get("episode_dir"):
        print(f"📁 {info['episode_dir']}")


def _wait_for_recording_to_start(node: RecordCLI, timeout_sec: float = 5.0) -> bool:
    if node._goal_started_evt.wait(timeout=timeout_sec):
        return True
    if node._goal_rejected_evt.is_set():
        return False
    node.get_logger().warning("Timed out waiting for recorder goal acceptance.")
    return False


def _finish_episode_or_warn(node: RecordCLI) -> bool:
    node.cancel_recording()
    if node._episode_finished_evt.wait(timeout=15.0):
        return True
    node.get_logger().warning(
        "Timed out waiting for recorder to finish. The recording server may have crashed — check its logs."
    )
    print(
        "⚠️  Recorder finalization did not complete in time. "
        "Discard/retry is disabled for this episode because writer state is unknown."
    )
    return False


def cli_loop(node):
    """Run the interactive prompt in a separate thread."""

    print("\nFetching dataset configuration from server...")
    if node._info_client.wait_for_service(timeout_sec=3.0):
        future = node._info_client.call_async(Trigger.Request())
        import time

        while rclpy.ok() and not future.done():
            time.sleep(0.1)
        if future.done() and future.result() is not None:
            try:
                import json

                info = json.loads(future.result().message)
                path = info.get("path", "Unknown")
                count = info.get("episodes", 0)

                print("\n========================================")
                print("📊 DATASET TARGET INFO")
                print("========================================")
                print(f"📁 Path: {path}")
                if count > 0:
                    print(f"⚠️  Found {count} existing episodes in this directory.")
                    print("   New recordings will be APPENDED to this dataset.")
                else:
                    print("✨ New dataset directory. No existing data found.")
                print("")
                print("💡 Tip: To change the dataset name, restart the launch server with:")
                print("   ros2 launch ... dataset_name:=<new_name> bag_base_dir:=<custom_dir>")
                print("========================================")

                ans = input("\nPress Enter to CONFIRM and continue, or 'q' to quit > ")
                if ans.strip().lower() in ["q", "quit"]:
                    rclpy.shutdown()
                    return
            except Exception as e:
                print(f"Failed to parse server info: {e}")
    else:
        print("⚠️  Warning: Could not fetch dataset info from server (timeout).")

    last_prompt = "default_task"

    while rclpy.ok():
        node._episode_finished_evt.clear()

        print("\n========================================")
        print("Dataset Collection CLI")
        print(f"Enter prompt text to start recording. (Press Enter to reuse: '{last_prompt}')")
        print("Type 'q' or 'quit' to exit.")
        print("========================================")

        try:
            prompt = input("Prompt > ")
            if prompt.strip().lower() in ["q", "quit"]:
                print("Exiting...")
                rclpy.shutdown()
                break

            if not prompt.strip():
                prompt = last_prompt
            else:
                last_prompt = prompt.strip()

            while rclpy.ok():
                node._episode_finished_evt.clear()
                node.send_goal(prompt)
                if not _wait_for_recording_to_start(node):
                    break

                recording_action = _recording_command(input())
                if not _finish_episode_or_warn(node):
                    break

                last_info = node.get_last_episode_info()
                can_modify_episode = node._last_result_success and last_info is not None
                if node._last_result_success and last_info is None:
                    print("⚠️  Recorder finalized, but episode details are unavailable; discard/retry is disabled.")
                elif not node._last_result_success:
                    print(
                        "⚠️  Recording did not finalize successfully; discard/retry is disabled "
                        "to avoid deleting an earlier episode."
                    )

                if recording_action == "discard":
                    if can_modify_episode:
                        node.delete_last_episode()
                    else:
                        print("⚠️  No current finalized episode is available to discard.")
                    break
                if recording_action == "retry":
                    if can_modify_episode and node.delete_last_episode():
                        print(f"🔁 Retrying with same prompt: {prompt}")
                        continue
                    if not can_modify_episode:
                        print("⚠️  No current finalized episode is available to discard before retry.")
                    break

                _print_episode_review(last_info, prompt)
                while rclpy.ok():
                    choice = _review_choice(input("Choice [s] > "))
                    if choice is None:
                        print("Unknown choice. Use Enter/s=save, d=discard, r=retry, q=quit and keep.")
                        continue
                    if choice == "save":
                        _print_kept_episode(last_info)
                        break
                    if choice == "discard":
                        if can_modify_episode:
                            node.delete_last_episode()
                        else:
                            print("⚠️  No current finalized episode is available to discard.")
                        break
                    if choice == "retry":
                        if can_modify_episode and node.delete_last_episode():
                            print(f"🔁 Retrying with same prompt: {prompt}")
                            break
                        if not can_modify_episode:
                            print("⚠️  No current finalized episode is available to discard before retry.")
                        choice = "discard_failed"
                        break
                    if choice == "quit_keep":
                        _print_kept_episode(last_info)
                        print("Exiting...")
                        rclpy.shutdown()
                        return

                if choice == "retry":
                    continue
                break

        except EOFError:
            break
        except Exception:
            traceback.print_exc()


def main(args=None):
    rclpy.init(args=args)
    node = RecordCLI()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    # Run ROS spinning in a background thread so input() doesn't block callbacks
    spin_thread = threading.Thread(target=executor.spin)
    spin_thread.start()

    try:
        # Run the interactive CLI in the main thread
        cli_loop(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join()


if __name__ == "__main__":
    main()
