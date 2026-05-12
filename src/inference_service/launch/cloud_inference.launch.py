from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    policy_path_arg = DeclareLaunchArgument(
        'policy_path',
        description='Absolute path to the pretrained policy model directory'
    )
    
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='auto',
        description='Device to run inference on (cuda, cpu, npu, ascend_om, ascend_om_3403, auto)'
    )
    
    input_topic_arg = DeclareLaunchArgument(
        'input_topic',
        default_value='/preprocessed/batch',
        description='Topic to subscribe for preprocessed batches from Edge'
    )
    
    output_topic_arg = DeclareLaunchArgument(
        'output_topic',
        default_value='/inference/action',
        description='Topic to publish inference results back to Edge'
    )

    cloud_node = Node(
        package='inference_service',
        executable='pure_inference_node',
        name='pure_inference_cloud',
        output='screen',
        parameters=[{
            'policy_path': LaunchConfiguration('policy_path'),
            'device': LaunchConfiguration('device'),
            'input_topic': LaunchConfiguration('input_topic'),
            'output_topic': LaunchConfiguration('output_topic'),
            # In a real cloud environment without simulation, use_sim_time should default to false
            'use_sim_time': False,
        }]
    )

    return LaunchDescription([
        policy_path_arg,
        device_arg,
        input_topic_arg,
        output_topic_arg,
        cloud_node
    ])
