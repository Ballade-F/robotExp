#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "map.hpp"
#include "message/msg/env_state.hpp"
#include "message/msg/robot_ctrl.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "uvs_message/msg/uv_opt_pose_list.hpp"

#include "json/json.h"

using namespace std::chrono_literals;
using std::placeholders::_1;

class ExpServer : public rclcpp::Node
{
public:
    
    //map
    int map_Nrobot;
    int map_Ntask;
    int map_Nobstacle;
    int map_ob_points;
    double map_resolution_x;
    double map_resolution_y;
    int map_Nx;
    int map_Ny;

    vector<Vector3d> robot_states;//各个机器人当前的状态 x, y, theta
    vector<Vector3d> task_states;//各个任务的位置 x, y, theta 但是theta不用
    vector<uint8_t> task_finished;//各个任务是否完成

    std::shared_ptr<Map_2D> map_ptr;
    double dt = 0.1;
    double finish_radius = 0.3;

    //rviz2
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_grid_publisher_;
    std::shared_ptr<nav_msgs::msg::OccupancyGrid> occupancy_grid_msg;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_array_publisher_;
    std::shared_ptr<visualization_msgs::msg::MarkerArray> marker_array_msg;

    //0.1s发送一次消息，各个robot和task的最新状态
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<message::msg::EnvState>::SharedPtr publisher_;
    //接收动捕消息
    rclcpp::Subscription<uvs_message::msg::UvOptPoseList>::SharedPtr subscription_;


    ExpServer();
    void timer_callback();
    void env_callback(const uvs_message::msg::UvOptPoseList::SharedPtr msg);
    
    void csv2vector(const string& csv_path, vector<vector<Vector2d>>& obstacles_, int n_robot, int n_task, int n_obstacle, int ob_point);

    void publishOccupancyGrid();
    void publishRobotTaskStates();
};