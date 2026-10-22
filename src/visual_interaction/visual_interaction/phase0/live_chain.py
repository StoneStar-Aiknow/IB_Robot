"""Run the whole chain live and print each round: camera, VL, checker, decision.

For bring-up on the board. Starts the real node against the real camera and
the real model, and prints what every round decided as it happens.

Why it supplies its own ActorState: the session refuses to spend a model call
without an unambiguous target, so with no actor publisher every round
short-circuits to STALE and the VL is never reached. The robot's actor
publisher is ``motion_imitation_stream``, which needs the YOLOX and PEAR model
services up; until those run, ``--synthetic-actor`` stands in. That is a
scaffold, not perception: it claims a visitor is present in the middle of the
frame and never changes, so nothing it produces says anything about actor
association, epoch bumps, or the ambiguity and hand-over paths. Point
``--actor-topic`` at the real stream instead once it is running.

Usage (board, with the environment loaded as ibrobot-worktree-env describes):
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    # aclruntime is a CANN binary extension installed into the SYSTEM
    # site-packages, which a venv does not inherit. Without this the model
    # loads nowhere and every round reports MODEL_NOT_READY with
    # ModuleNotFoundError: No module named 'aclruntime'.
    export PYTHONPATH=/usr/local/lib64/python3.11/site-packages:$PYTHONPATH
    python3 -m visual_interaction.phase0.live_chain \\
        --demo-root /root/vlx_internvl_310p --synthetic-actor --seconds 120

Note on --seconds: the session's duration limit (session.max_session_sec,
120 s by default) is a missed-hand-over signal that only an attendant clears,
and a synthetic actor never leaves. A run longer than that limit therefore
spends its remainder reporting WAITING_NEXT_VISITOR. Keep a synthetic run
inside the limit, or drive the real actor stream.

Prints one line per round. A spoken round shows the text; a quiet round shows
its reason. Nothing is ever spoken aloud — the utterance goes to the topic.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time

CAMERA_TOPIC = "/aima/hal/sensor/rgb_head_front_center/rgb_image_half/compressed"
CAMERA_QOS = {"reliability": "best_effort", "durability": "volatile", "history": "keep_last", "depth": 5}
SYNTHETIC_ACTOR_TOPIC = "/live_chain/actor_state"


def _synthetic_actor_node(topic: str, rate_hz: float, width: int, height: int):
    """A node claiming one visitor, centred and still. Scaffolding only."""
    import rclpy

    from ibrobot_msgs.msg import ActorState

    node = rclpy.create_node("live_chain_synthetic_actor")
    publisher = node.create_publisher(ActorState, topic, 10)
    # Centre half of the frame, as [x_min, y_min, x_max, y_max] source pixels.
    box = [width * 0.25, height * 0.25, width * 0.75, height * 0.75]

    def tick() -> None:
        now = node.get_clock().now().to_msg()
        message = ActorState()
        # header.stamp is the observation time of the frame the decision is
        # for; detections_stamp is the YOLOX result it rests on. Both are
        # "now" here, which is the one way this scaffold flatters itself: a
        # real stream's detections_stamp trails the frame.
        message.header.stamp = now
        message.detections_stamp = now
        message.persons_count = 1
        message.actor_present = True
        message.start_ready = True
        message.actor_id = 0
        message.bbox = [float(value) for value in box]
        message.confidence = 0.9
        # A value the contract defines: the session reads "no_detections" and
        # "stale_detections" as unknown, so an invented reason would be taken
        # as a known answer by accident rather than on purpose.
        message.reason = "highest_confidence_person"
        publisher.publish(message)

    node.create_timer(1.0 / rate_hz, tick)
    return node


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo-root", default="", help="board InternVL deployment; omit to run the fake model")
    parser.add_argument("--camera-topic", default=CAMERA_TOPIC)
    parser.add_argument("--actor-topic", default="", help="the real actor stream; omit with --synthetic-actor")
    parser.add_argument("--synthetic-actor", action="store_true", help="stand in for perception (scaffold)")
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--round-period", type=float, default=2.0)
    parser.add_argument("--frames-per-window", type=int, default=2)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--min-gap-sec", type=float, default=5.0, help="quiet time after speaking")
    args = parser.parse_args(argv)

    if bool(args.actor_topic) == bool(args.synthetic_actor):
        print("live_chain: pass exactly one of --actor-topic or --synthetic-actor", file=sys.stderr)
        return 2

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.parameter import Parameter

    from visual_interaction.visual_interaction_node import VisualInteractionNode

    actor_topic = args.actor_topic or SYNTHETIC_ACTOR_TOPIC
    overrides = [
        Parameter("runtime_description_json", value=json.dumps({"robot": {}, "interfaces": {}})),
        Parameter("camera_topic", value=args.camera_topic),
        Parameter("camera_topic_qos_json", value=json.dumps(CAMERA_QOS)),
        Parameter("categories_json", value='["toy","sign","hat","badge"]'),
        Parameter("actor_state_topic", value=actor_topic),
        Parameter("round_period_sec", value=float(args.round_period)),
        Parameter("window_sec", value=float(args.window_sec)),
        Parameter("frames_per_window", value=int(args.frames_per_window)),
        Parameter("min_gap_sec", value=float(args.min_gap_sec)),
        Parameter("status_period_sec", value=10.0),
    ]
    if args.demo_root:
        overrides += [
            Parameter("vl_implementation", value="internvl"),
            Parameter("vl_demo_root", value=args.demo_root),
        ]

    rclpy.init()
    executor = SingleThreadedExecutor()
    nodes = []
    try:
        if args.synthetic_actor:
            # 1344x972 is the relay's frame size; the box only has to be a
            # plausible, stable region of it.
            actor = _synthetic_actor_node(SYNTHETIC_ACTOR_TOPIC, 10.0, 1344, 972)
            nodes.append(actor)
            executor.add_node(actor)
            print(f"live_chain: synthetic actor on {SYNTHETIC_ACTOR_TOPIC} (scaffold, not perception)")

        print(f"live_chain: loading{' the model (~20 s)' if args.demo_root else ''}...", flush=True)
        node = VisualInteractionNode(parameter_overrides=overrides)
        nodes.append(node)
        executor.add_node(node)

        # The node logs a spoken round at info and a quiet one at debug; for a
        # bring-up run every round is worth seeing.
        import rclpy.logging

        rclpy.logging.set_logger_level(node.get_name(), rclpy.logging.LoggingSeverity.DEBUG)

        # And print every round through Python as well, because the model's own
        # loader silences the ROS log. aclruntime writes ~20 debug lines per
        # inference straight to fd 1, which log_level cannot suppress, so the
        # demo's quiet_native_output points fd 1 and 2 at /dev/null and rebinds
        # only Python's sys.stdout object. ROS logging writes to the fd from C,
        # so it goes to /dev/null with the noise — the first live run produced
        # 54 rounds and not one visible line.
        #
        # Wrapping _publish rather than the on_round hook: the hook is handed
        # the reason code alone, and what a bring-up run needs is the spoken
        # text and the rejected answer's detail.
        published = node._publish

        def echo(result, window):
            published(result, window)
            if result.spoken:
                observation = result.observation
                subject = f"{observation.category}/{observation.region}" if observation else "?"
                print(f'  SAY: "{result.text}"  [{subject}]', flush=True)
            elif result.detail:
                print(f"  quiet: {result.reason} ({result.detail})", flush=True)
            else:
                # MODEL_NOT_READY keeps its cause on the port, not the result.
                cause = getattr(node._session.hooks.vision, "last_error", "")
                print(f"  quiet: {result.reason}{f' ({cause})' if cause else ''}", flush=True)

        node._publish = echo

        if args.demo_root:
            # Pay the load cost before the first round so the first round costs
            # what every later one does.
            vision = node._session.hooks.vision
            loader = getattr(vision, "load", None)
            if loader is not None:
                started = time.monotonic()
                try:
                    loader()
                    print(f"live_chain: model ready in {time.monotonic() - started:.1f}s", flush=True)
                except Exception as error:  # noqa: BLE001 - reported, the run continues
                    print(f"live_chain: model NOT ready: {error}", file=sys.stderr, flush=True)

        print(f"live_chain: running {args.seconds:g}s, one round every {args.round_period:g}s", flush=True)
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        print("live_chain: stopped")
    finally:
        # Same lesson pose_recorder learned, applied late: Ctrl-C has already
        # shut the context down, so rclpy.shutdown() raises "rcl_shutdown
        # already called" and anything after it is skipped. The summary is the
        # run's result, so it is gathered first and printed regardless, and
        # every teardown call is suppressed rather than trusted.
        status = None
        for item in nodes:
            if hasattr(item, "_session"):
                status = {
                    "rounds": item._rounds,
                    "frames": item._buffer.total,
                    "outcomes": dict(sorted(item._counts.items())),
                    "spoken_topics": item._session.ledger().spoken_count(),
                }
        if status:
            print(f"live_chain: {json.dumps(status, ensure_ascii=False)}", flush=True)
        for item in nodes:
            with contextlib.suppress(Exception):
                executor.remove_node(item)
            with contextlib.suppress(Exception):
                item.destroy_node()
        with contextlib.suppress(Exception):
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
