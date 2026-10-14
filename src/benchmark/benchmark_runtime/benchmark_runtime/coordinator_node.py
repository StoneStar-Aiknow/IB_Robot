"""Bounded static fan-in coordinator. No simulator or model instance is created."""

from __future__ import annotations

import contextlib
import math
import time
from pathlib import Path

import rclpy
import yaml
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from benchmark_runtime.canonical_report import CanonicalReportWriter
from benchmark_runtime.lane_aggregation import global_plan_from_manifest, read_object, write_global_report


class BenchmarkCoordinatorNode(Node):
    """Launch once per global run; exit only after the global terminal report."""

    def __init__(self, parameter_overrides=None):
        super().__init__("benchmark_coordinator", parameter_overrides=parameter_overrides)
        self.declare_parameter("robot_config_path", "")
        self.declare_parameter("run_id", "")
        self.declare_parameter("lane_count", 1)
        # Zero means snapshot immediately, intended for launch after all evaluator exits.
        self.declare_parameter("wait_timeout_sec", 3600.0)
        self.terminal = False
        self.exit_code = 1
        self._plan = None

    def start(self):
        path = Path(self.get_parameter("robot_config_path").value).expanduser().resolve()
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        robot = raw.get("robot", raw)
        evaluation = robot["benchmark"]["evaluation"]
        parallel = evaluation.get("parallel", {})
        if parallel.get("scheduling", "static") != "static" or parallel.get("fail_fast", True) is not True:
            raise ValueError("coordinator supports static/fail_fast=true only")
        self._run_id = self.get_parameter("run_id").value
        if not self._run_id or self._run_id in {".", ".."} or Path(self._run_id).name != self._run_id:
            raise ValueError("coordinator requires a shared run_id (single path component)")
        self._lane_count = self.get_parameter("lane_count").value
        if type(self._lane_count) is not int or self._lane_count < 2:
            raise ValueError("coordinator lane_count must be >= 2; single lane needs no coordinator")
        active = parallel.get("lane_count", 1) if parallel.get("enabled", False) else 1
        if active != self._lane_count:
            raise ValueError("coordinator lane_count must match active evaluation.parallel.lane_count")
        if not evaluation.get("output", {}).get("write_canonical", True):
            raise ValueError("coordinator requires output.write_canonical=true")
        timeout = self.get_parameter("wait_timeout_sec").value
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("wait_timeout_sec must be finite and nonnegative")
        output = Path(evaluation.get("output", {}).get("root", "outputs/benchmark")).expanduser()
        if not output.is_absolute():
            workspace = next((parent for parent in path.parents if (parent / ".git").exists()), path.parent)
            output = workspace / output
        self._root = output.resolve() / self._run_id
        if any((self._root / name).exists() for name in ("run_manifest.json", ".aggregation_claim")):
            raise ValueError("global report already exists; use a fresh run_id")
        self._evaluation = evaluation
        self._deadline = time.monotonic() + timeout
        self._timer = self.create_timer(0.1, self._poll)

    def _poll(self):
        if self.terminal:
            return
        statuses = []
        for index in range(self._lane_count):
            try:
                manifest = read_object(self._root / "lanes" / f"lane_{index:03d}" / "run_manifest.json")
                if manifest.get("run_id") != self._run_id:
                    raise ValueError("lane manifest run_id mismatch")
                plan = global_plan_from_manifest(manifest)
                evaluation = self._evaluation
                if (
                    plan.suite != evaluation["suite"]
                    or plan.seed != evaluation.get("seed", 10000)
                    or plan.episodes_per_task != evaluation.get("episodes_per_task", 10)
                    or plan.max_steps != evaluation.get("max_steps", 600)
                    or plan.lane_count != self._lane_count
                    or plan.requested_task_order_index != evaluation.get("task_order_index", 0)
                    or (evaluation.get("tasks") is None and plan.selected_task_ids != tuple(range(len(plan.tasks))))
                    or (evaluation.get("tasks") is not None and plan.selected_task_ids != tuple(evaluation["tasks"]))
                ):
                    raise ValueError("lane global plan does not match coordinator configuration")
                if self._plan is None:
                    self._plan = plan
                elif plan != self._plan:
                    raise ValueError("lane global plans disagree")
                statuses.append(manifest.get("status"))
            except FileNotFoundError:
                statuses.append("missing")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.get_logger().error(str(exc))
                statuses.append("failed")
        if (
            all(status in {"complete", "partial", "failed"} for status in statuses)
            or any(status in {"partial", "failed"} for status in statuses)
            or time.monotonic() >= self._deadline
        ):
            self.finish("lane failure or deadline" if "complete" not in statuses else "fan-in")

    def finish(self, reason):
        if self.terminal:
            return
        self.terminal = True
        try:
            if self._plan is None:
                # No provider snapshot exists: explicitly unknown, never a fabricated plan.
                writer = CanonicalReportWriter(self._root)
                writer.start_run({"run_id": self._run_id, "status": "running"}, allow_lanes=True)
                writer.write_summary(
                    {
                        "schema_version": 1,
                        "run_id": self._run_id,
                        "status": "partial",
                        "run_status": "partial",
                        "suite": self._evaluation["suite"],
                        "counters": None,
                        "metadata": {"error": reason, "plan_unavailable": True, "lane_count": self._lane_count},
                    }
                )
                writer.finalize_manifest({"run_id": self._run_id, "status": "partial"})
                status = "partial"
            else:
                summary = write_global_report(
                    self._root, run_id=self._run_id, plan=self._plan, lane_count=self._lane_count
                )
                status = summary.run_status
            self.exit_code = 0 if status == "complete" else 1
            self.get_logger().info(
                f"[IBROBOT_BENCHMARK][GLOBAL_EVALUATION_FINISHED] status={status} "
                f"run_id={self._run_id} summary={self._root / 'summary.json'}"
            )
        except Exception as exc:
            self.exit_code = 1
            self.get_logger().error(f"global report write failed: {exc}")


def main(argv=None):
    rclpy.init(args=argv)
    node = BenchmarkCoordinatorNode()
    try:
        node.start()
        while rclpy.ok() and not node.terminal:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not node.terminal:
            node.finish("coordinator ROS shutdown")
    except (KeyboardInterrupt, ExternalShutdownException):
        if hasattr(node, "_root"):
            node.finish("coordinator interrupted")
    except Exception as exc:
        node.get_logger().error(str(exc))
    finally:
        node.destroy_node()
        with contextlib.suppress(Exception):
            rclpy.shutdown()
    if node.exit_code:
        raise SystemExit(node.exit_code)


if __name__ == "__main__":
    main()
