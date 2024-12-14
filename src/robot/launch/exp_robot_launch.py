from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
from launch.actions import ExecuteProcess

def generate_launch_description():
    node_name = LaunchConfiguration('node_name')
    node_name_launch_arg = DeclareLaunchArgument(
        'node_name',
        default_value='robot_0_node'
    )
    package_name = 'robot'
    package_share_dir = get_package_share_directory(package_name)
    config = os.path.join(package_share_dir, 'config', 'exp_launch.yaml')
    robot_node = Node(
        package="robot",
        executable="robot_exp_node",
        name=node_name,
        output='screen',
        parameters=[config]
    )
    
    ld = LaunchDescription(
        [
            node_name_launch_arg,
            robot_node
        ]
    )
   
    return ld