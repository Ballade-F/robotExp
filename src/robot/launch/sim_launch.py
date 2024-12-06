from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    package_name = 'robot'
    package_share_dir = get_package_share_directory(package_name)
    config = os.path.join(package_share_dir, 'config', 'sim_launch.yaml')
    robot_0_node = Node(
        package="robot",
            executable="robot_node",
            name='robot_0_node',
            output='screen',
            parameters=[config]
        )
    sim_node = Node(
        package="robot",
        executable="sim_node",
        name='sim_node',
        output='screen',
        parameters=[config]
    )
    rviz_config = os.path.join(
      package_share_dir,
      'rviz2',
      'sim.rviz'
      )
    rviz2_node = Node(
        package="rviz2",
        executable="rviz2",
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config]
    )
    
    ld = LaunchDescription(
        [
            robot_0_node,
            sim_node,
            rviz2_node
        ]
    )
   
    
    return ld