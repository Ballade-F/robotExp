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

#include "json/json.h"

using namespace std::chrono_literals;
using std::placeholders::_1;

class SimServer : public rclcpp::Node
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
    vector<Vector2d> robot_ctrls;//各个机器人当前的控制 v, w
    vector<Vector3d> task_states;//各个任务的位置 x, y, theta 但是theta不用
    vector<uint8_t> task_finished;//各个任务是否完成
    vector<uint8_t> robot_com_count;//机器人通信计数器
    uint8_t perception_max = 5; //感知丢失最大数

    std::shared_ptr<Map_2D> map_ptr;
    double dt = 0.1;
    double finish_radius = 0.3;

    //rviz2
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_grid_publisher_;
    std::shared_ptr<nav_msgs::msg::OccupancyGrid> occupancy_grid_msg;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_array_publisher_;
    std::shared_ptr<visualization_msgs::msg::MarkerArray> marker_array_msg;



    SimServer(const vector<Vector3d> &start_, const vector<Vector3d> &task_);
    void updateStates();

    // //显示与画图用
    // vector<vector<Vector3d>> robot_traces; // 各个机器人轨迹，最外层为帧数，中间层为机器人，最内层为坐标
    // vector<vector<uint8_t>> task_traces; // 各个任务在各个帧数是否完成
    
    //0.1s发送一次消息，各个robot和task的最新状态
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<message::msg::EnvState>::SharedPtr publisher_;
    //接受各个机器人的u
    rclcpp::Subscription<message::msg::RobotCtrl>::SharedPtr subscription_;
    int64_t time_count_;
    void timer_callback();
    void robot_u_callback(const message::msg::RobotCtrl::SharedPtr msg);
    
    void csv2vector(const string& csv_path, vector<vector<Vector2d>>& obstacles_, int n_robot, int n_task, int n_obstacle, int ob_point);

    void publishOccupancyGrid();
    void publishRobotTaskStates();

};