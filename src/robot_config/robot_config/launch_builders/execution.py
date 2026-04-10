"""Execution system launch builders (refactored).

This module handles:
- Inference service node generation (with contract synthesis)
- Action dispatcher node generation
- Automatic parameter binding from robot_config
- Support for monolithic and distributed (cloud-edge) modes

Execution Modes:
- monolithic: All inference in one process (zero-copy, default)
- distributed: Edge preprocessing → Cloud inference → Edge postprocessing
"""

import os
from launch_ros.actions import Node

from robot_config.utils import parse_bool, prepare_lerobot_env


def generate_inference_node(robot_config, control_mode, use_sim=False,
                            cloud_local=False):
    """Generate inference service node with auto-synthesized contract.

    This function:
    1. Synthesizes contract from robot_config
    2. Saves contract for debugging and inference_service
    3. Creates inference node with proper parameters

    Args:
        robot_config: Robot configuration dict
        control_mode: Active control mode
        use_sim: Simulation mode flag
        cloud_local: In distributed mode, also launch cloud node locally

    Returns:
        Node action for inference service, or None if not enabled

    Raises:
        ContractSynthesisError: If configuration is architecturally invalid
    """
    is_sim = parse_bool(use_sim, default=False)

    control_modes = robot_config.get("control_modes", {})
    if control_mode not in control_modes:
        print(f"[robot_config] WARNING: Control mode '{control_mode}' not found")
        return None

    mode_config = control_modes[control_mode]
    inference_config = mode_config.get("inference", {})

    if not inference_config.get("enabled", False):
        print(f"[robot_config] Inference not enabled for mode '{control_mode}'")
        return None

    execution_mode = inference_config.get("execution_mode", "monolithic")

    if execution_mode == "distributed":
        return generate_distributed_inference_nodes(
            robot_config, control_mode, use_sim, cloud_local=cloud_local)

    return generate_monolithic_inference_node(robot_config, control_mode, use_sim)


def generate_monolithic_inference_node(robot_config, control_mode, use_sim=False):
    """Generate monolithic (self-contained) inference node."""
    is_sim = parse_bool(use_sim, default=False)

    control_modes = robot_config.get("control_modes", {})
    mode_config = control_modes[control_mode]
    inference_config = mode_config.get("inference", {})

    execution_mode = inference_config.get("execution_mode", "monolithic")
    request_timeout = inference_config.get("request_timeout", 5.0)
    cloud_inference_topic = inference_config.get("cloud_inference_topic", "/preprocessed/batch")
    cloud_result_topic = inference_config.get("cloud_result_topic", "/inference/action")

    print(
        f"[robot_config] ========== Generating Inference Node (Monolithic) =========="
    )
    print(f"[robot_config] Control mode: {control_mode}")
    print(f"[robot_config] Execution mode: {execution_mode}")

    robot_config_path = robot_config.get('_config_path', '')
    if not robot_config_path:
        raise ValueError(
            "robot_config dict is missing '_config_path'. Ensure loader.py injects this correctly."
        )

    model_name = inference_config["model"]
    models = robot_config.get("models", {})
    if model_name not in models:
        print(f"[robot_config] ERROR: Model '{model_name}' not found in config")
        return None

    model_config = models[model_name]

    print(f"[robot_config] Model: {model_name}")
    print(f"[robot_config]   Path: {model_config['path']}")
    print(f"[robot_config]   Policy type: {model_config.get('policy_type', 'unknown')}")
    print(f"[robot_config]   Robot config: {robot_config_path}")

    env = prepare_lerobot_env()
    if env.get("PYTHONPATH"):
        print(f"[robot_config] Injected PYTHONPATH: {env['PYTHONPATH']}")

    node_params = {
        "checkpoint": model_config["path"],
        "robot_config_path": str(robot_config_path),
        "lerobot_norm_mode": model_config.get("lerobot_norm_mode", "range_m100_100"),
        "passive_mode": True,
        "device": "auto",
        "use_sim_time": is_sim,
        "node_name": "act_inference_node",
        "execution_mode": execution_mode,
        "request_timeout": request_timeout,
        "cloud_inference_topic": cloud_inference_topic,
        "cloud_result_topic": cloud_result_topic,
    }

    inference_node = Node(
        package="inference_service",
        executable="lerobot_policy_node",
        name="act_inference_node",
        env=env,
        parameters=[node_params],
        output="screen",
    )

    print(f"[robot_config] ✓ Monolithic inference node configured")
    return inference_node


def generate_distributed_inference_nodes(robot_config, control_mode, use_sim=False,
                                         cloud_local=False):
    """Generate distributed inference nodes.

    By default only the edge proxy node is generated here. The cloud
    (pure_inference_node) is intended to run on a separate GPU machine
    via ``cloud_inference.launch.py``.

    Set *cloud_local=True* to co-locate both nodes for single-machine
    testing / debugging.

    Args:
        robot_config: Robot configuration dict
        control_mode: Active control mode
        use_sim: Simulation mode flag
        cloud_local: If True, also launch cloud node locally (single-machine test)

    Returns:
        List of Node actions
    """
    is_sim = parse_bool(use_sim, default=False)

    control_modes = robot_config.get("control_modes", {})
    mode_config = control_modes[control_mode]
    inference_config = mode_config.get("inference", {})

    request_timeout = inference_config.get("request_timeout", 5.0)
    cloud_inference_topic = inference_config.get("cloud_inference_topic", "/preprocessed/batch")
    cloud_result_topic = inference_config.get("cloud_result_topic", "/inference/action")

    print(
        f"[robot_config] ========== Generating Inference Nodes (Distributed) =========="
    )
    print(f"[robot_config] Control mode: {control_mode}")
    print(f"[robot_config] Architecture: Edge Proxy + Cloud Inference")
    print(f"[robot_config] Cloud co-located: {cloud_local}")

    robot_config_path = robot_config.get('_config_path', '')
    if not robot_config_path:
        raise ValueError(
            "robot_config dict is missing '_config_path'. Ensure loader.py injects this correctly."
        )

    model_name = inference_config["model"]
    models = robot_config.get("models", {})
    if model_name not in models:
        print(f"[robot_config] ERROR: Model '{model_name}' not found in config")
        return None

    model_config = models[model_name]
    policy_path = model_config["path"]

    print(f"[robot_config] Model: {model_name}")
    print(f"[robot_config]   Path: {policy_path}")
    print(f"[robot_config]   Policy type: {model_config.get('policy_type', 'unknown')}")
    print(f"[robot_config]   Robot config: {robot_config_path}")

    env = prepare_lerobot_env()

    nodes = []

    # --- Edge node: always launched locally ---
    edge_node_params = {
        "checkpoint": policy_path,
        "robot_config_path": str(robot_config_path),
        "lerobot_norm_mode": model_config.get("lerobot_norm_mode", "range_m100_100"),
        "passive_mode": True,
        "device": "auto",
        "use_sim_time": is_sim,
        "node_name": "act_inference_node",
        "execution_mode": "distributed",
        "request_timeout": request_timeout,
        "cloud_inference_topic": cloud_inference_topic,
        "cloud_result_topic": cloud_result_topic,
    }

    edge_node = Node(
        package="inference_service",
        executable="lerobot_policy_node",
        name="act_inference_node",
        env=env,
        parameters=[edge_node_params],
        output="screen",
    )
    nodes.append(edge_node)
    print(f"[robot_config]   Edge Node (lerobot_policy_node): Action Server + Pre/Post processing")
    print(f"[robot_config]     Publishing to: {cloud_inference_topic}")
    print(f"[robot_config]     Subscribed to: {cloud_result_topic}")

    # --- Cloud node: only launched locally when cloud_local=True ---
    if cloud_local:
        cloud_node_params = {
            "policy_path": policy_path,
            "input_topic": cloud_inference_topic,
            "output_topic": cloud_result_topic,
            "device": "auto",
            "use_sim_time": is_sim,
        }

        cloud_node = Node(
            package="inference_service",
            executable="pure_inference_node",
            name="pure_inference",
            env=env,
            parameters=[cloud_node_params],
            output="screen",
        )
        nodes.append(cloud_node)
        print(f"[robot_config]   Cloud Node (pure_inference_node): GPU Inference (co-located)")
        print(f"[robot_config]     Subscribed to: {cloud_inference_topic}")
        print(f"[robot_config]     Publishing to: {cloud_result_topic}")
    else:
        print(f"[robot_config]   ⚠ Cloud node NOT launched locally.")
        print(f"[robot_config]     Launch it on the GPU machine with:")
        print(f"[robot_config]       ros2 launch inference_service cloud_inference.launch.py \\")
        print(f"[robot_config]         policy_path:={policy_path} device:=cuda")

    print(
        f"[robot_config] ✓ Distributed inference nodes configured ({len(nodes)} nodes)"
    )
    return nodes


def generate_action_dispatcher_node(robot_config, control_mode, use_sim=False):
    """Generate action dispatcher node with configuration binding.

    Args:
        robot_config: Robot configuration dict
        control_mode: Active control mode
        use_sim: Simulation mode flag

    Returns:
        Node action for action_dispatcher
    """
    is_sim = parse_bool(use_sim, default=False)

    robot_config_path = robot_config.get('_config_path', '')
    if not robot_config_path:
        raise ValueError(
            "robot_config dict is missing '_config_path'. Ensure loader.py injects this correctly."
        )

    control_modes = robot_config.get("control_modes", {})
    mode_config = control_modes.get(control_mode, {})
    executor_config = mode_config.get("executor", {})

    robot_name = robot_config.get("name", "so101")
    robot_joints = robot_config.get("joints", {})
    all_joints = robot_joints.get("all", ["1", "2", "3", "4", "5", "6"])

    executor_type = executor_config.get("type", "topic")
    executor_mode = executor_config.get("mode", control_mode)

    print(f"[robot_config] ========== Generating Action Dispatcher ==========")
    print(f"[robot_config] Robot: {robot_name}")
    print(f"[robot_config] Control mode: {control_mode}")
    print(f"[robot_config] Executor type: {executor_type}")
    print(f"[robot_config] Use sim time: {is_sim}")

    inference_config = mode_config.get("inference", {})

    action_server = "/act_inference_node/DispatchInfer"

    action_dispatcher_node = Node(
        package="action_dispatch",
        executable="action_dispatcher_node",
        name="action_dispatcher",
        parameters=[
            {
                "enable_dual_mode": executor_type == "topic",
                "executor_mode": executor_mode,
                "robot_name": robot_name,
                "joint_names": all_joints,
                "queue_size": executor_config.get("queue_size", 100),
                "watermark_threshold": executor_config.get("watermark_threshold", 20),
                "min_queue_size": executor_config.get("min_queue_size", 10),
                "control_frequency": executor_config.get("control_frequency", 100.0),
                "control_mode": control_mode,
                "interpolation_enabled": True,
                "interpolation_step": 0.1,
                "max_interpolation_time": 2.0,
                "on_inference_failure": "hold",
                "on_queue_exhausted": "hold",
                "max_inference_timeout": 1.0,
                "max_retry_attempts": 3,
                "retry_backoff_base": 0.5,
                "stale_obs_threshold_ms": 500,
                "exhaustion_timeout": 2.0,
                "joint_state_topic": "/joint_states",
                "dispatch_action_topic": "/action_dispatch/dispatch_action",
                "robot_config_path": str(robot_config_path),
                "inference_action_server": action_server,
                "inference_prompt": "",
                "use_sim_time": is_sim,
            }
        ],
        output="screen",
    )

    print(f"[robot_config] ✓ Action dispatcher configured")
    return action_dispatcher_node


def generate_execution_nodes(robot_config, control_mode="model_inference",
                             use_sim=False, cloud_local=False):
    """Generate all execution nodes (inference + dispatcher).

    This is the main entry point for execution system generation.
    It automatically determines whether to launch inference based on
    the control_mode configuration.

    Args:
        robot_config: Robot configuration dict
        control_mode: Active control mode (defaults to robot's default_control_mode)
        use_sim: Simulation mode flag
        cloud_local: In distributed mode, also launch cloud node locally

    Returns:
        List of Node actions for execution system
    """
    nodes = []

    if not control_mode or control_mode == "default":
        control_mode = robot_config.get("default_control_mode", "model_inference")

    try:
        inference_result = generate_inference_node(
            robot_config, control_mode, use_sim, cloud_local=cloud_local)
        if inference_result:
            if isinstance(inference_result, list):
                nodes.extend(inference_result)
            else:
                nodes.append(inference_result)
    except Exception as e:
        print(f"[robot_config] ERROR generating inference node: {e}")

    try:
        dispatcher_node = generate_action_dispatcher_node(
            robot_config, control_mode, use_sim
        )
        nodes.append(dispatcher_node)
    except Exception as e:
        print(f"[robot_config] ERROR generating action dispatcher: {e}")
        raise

    return nodes
