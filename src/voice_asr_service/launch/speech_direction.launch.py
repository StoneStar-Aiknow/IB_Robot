from collections.abc import Mapping
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_MODEL_PATH_KEYS = (
    "silero_vad_model_path",
    "fullsubnet_repo_dir",
    "fullsubnet_ckpt",
)


def _workspace_root() -> Path:
    """从 colcon 包安装前缀定位标准工作区根目录。"""
    package_prefix = Path(get_package_prefix("voice_asr_service")).resolve()
    # 标准独立安装为 <workspace>/install/<package>，merge-install 为 <workspace>/install。
    if package_prefix.parent.name == "install":
        return package_prefix.parent.parent
    if package_prefix.name == "install":
        return package_prefix.parent
    # 自定义 install base 无稳定的工作区反推规则；保留可用默认值，并允许 models_root 显式覆盖。
    return package_prefix.parent


def _load_speech_direction_parameters(config_path: str | Path, models_root: Path) -> dict[str, object]:
    """加载 voice 自有 YAML，并相对 models 根解析模型路径。"""
    config_file_path = Path(config_path)
    with config_file_path.open(encoding="utf-8") as config_file:
        document = yaml.safe_load(config_file)

    # 在 launch 边界拒绝畸形配置，避免节点收到难以定位的参数错误。
    if not isinstance(document, Mapping):
        raise ValueError(f"配置文件 {config_file_path}: YAML document 必须是 mapping")
    node_config = document.get("speech_direction_node")
    if not isinstance(node_config, Mapping):
        raise ValueError(f"配置文件 {config_file_path}: speech_direction_node 必须存在且为 mapping")
    parameter_config = node_config.get("ros__parameters")
    if not isinstance(parameter_config, Mapping):
        raise ValueError(f"配置文件 {config_file_path}: speech_direction_node.ros__parameters 必须存在且为 mapping")

    params = dict(parameter_config)
    for key in _MODEL_PATH_KEYS:
        value = params.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"配置文件 {config_file_path}: speech_direction_node.ros__parameters.{key} 必须是非空字符串"
            )
        path = Path(value.strip()).expanduser()
        if not path.is_absolute():
            path = models_root / path
        params[key] = str(path.resolve())
    return params


def _launch_setup(context):
    config_path = LaunchConfiguration("config_file").perform(context)
    models_root = Path(LaunchConfiguration("models_root").perform(context)).expanduser()
    params = _load_speech_direction_parameters(config_path, models_root)
    return [
        Node(
            package="voice_asr_service",
            executable="speech_direction_node",
            name="speech_direction_node",
            parameters=[params],
            output="screen",
        )
    ]


def generate_launch_description():
    default_config = str(Path(get_package_share_directory("voice_asr_service")) / "config" / "speech_direction.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="speech_direction ROS 参数 YAML",
            ),
            DeclareLaunchArgument(
                "models_root",
                default_value=str(_workspace_root() / "models"),
                description="模型根目录；YAML 中相对模型路径以此为基准",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
