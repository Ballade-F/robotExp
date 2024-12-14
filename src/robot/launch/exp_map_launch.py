from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
from launch.actions import ExecuteProcess

def generate_launch_description():
    package_name = 'robot'
    package_share_dir = get_package_share_directory(package_name)
    config = os.path.join(package_share_dir, 'config', 'exp_launch.yaml')
    exp_map_node = Node(
        package="robot",
        executable="exp_map_node",
        name='exp_map_node',
        output='screen',
        parameters=[config]
    )
    
    # uvs_optitrack
    uvs_optitrack = Node(
        package='uvs_optitrack',
        executable='uvs_optitrack',
        name='uvs_optitrack',
        output='screen',
        parameters=[config]
    )
    
    rviz_config = os.path.join(
      package_share_dir,
      'config',
      'rviz2',
      'exp_map.rviz'
      )
    print(rviz_config)
    rviz2_node = Node(
        package="rviz2",
        executable="rviz2",
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config]
    )
    
    ld = LaunchDescription(
        [
            uvs_optitrack,
            exp_map_node,
            rviz2_node
        ]
    )
   
    return ld