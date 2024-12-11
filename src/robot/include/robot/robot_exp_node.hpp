#pragma once
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "message/msg/env_state.hpp"
#include "message/msg/robot_ctrl.hpp"
#include "uvs_message/msg/uv_emb_status.hpp"
#include "uvs_message/msg/uv_emb_kinetics.hpp"

#include "json/json.h"
#include "robot.hpp"

using namespace std;
using namespace Eigen;
using namespace std::chrono_literals;
using std::placeholders::_1;

#define ROBOT_CONTROL_PERIOD (100ms)
#define ROBOT_KEYFRAME_PERIOD (1000ms)
#define ROBOT_DECISION_PERIOD (2000ms)
#define ROBOT_BUFFER_SIZE 5


class RobotExpNode : public rclcpp::Node
{
public:
    int robot_id;
    //map
    int map_Nrobot;
    int map_Ntask;
    int map_Nobstacle;
    int map_ob_points;
    double map_resolution_x;
    double map_resolution_y;
    int map_Nx;
    int map_Ny;
    //planner
    int planner_Ntheta;
    double planner_Vmax;
    double planner_Wmax;
    int planner_Vstep;
    int planner_Wstep;
    int planner_TraceStep;
    double planner_dt;
    double planner_Rfinish;
    //mpc
    int mpc_N;
    double mpc_dt;
    double mpc_wheel_Vmax;
    double mpc_wheel_width;
    double mpc_Qxy;
    double mpc_Qtheta;
    double mpc_Rv;
    double mpc_Rw;


    vector<Vector3d> robot_states;//各个机器人当前的状态 x, y, theta
    vector<Vector3d> task_states;//各个任务的位置 x, y, theta 但是theta不用
    vector<uint8_t> task_finished;//各个任务是否完成
    Vector2d self_speed;

    std::shared_ptr<Map_2D> map_p;
    std::shared_ptr<MPC> mpc_p;
    std::shared_ptr<HybridAStar> hybrid_astar_p;
    std::shared_ptr<HybridAStar> hybrid_dist_astar_p;
    std::shared_ptr<Network> network_p;

    std::shared_ptr<Robot> robot_p;

    RobotExpNode();
    // ~RobotNode();


    //订阅感知信息
    rclcpp::Subscription<message::msg::EnvState>::SharedPtr subscription_;
    //订阅轮速
    rclcpp::Subscription<uvs_message::msg::UvEmbStatus>::SharedPtr subscription_status;
    //发布控制信息
    rclcpp::Publisher<uvs_message::msg::UvEmbKinetics>::SharedPtr publisher_;
    //定时器 用于控制频率
    rclcpp::TimerBase::SharedPtr timer_ctrl;
    rclcpp::TimerBase::SharedPtr timer_keyframe;
    //定时器
    rclcpp::CallbackGroup::SharedPtr cb_group_decision;
    rclcpp::TimerBase::SharedPtr timer_decision;

    void ctrl_timer_callback();
    void decision_timer_callback();
    void keyframe_timer_callback();
    void env_callback(const message::msg::EnvState::SharedPtr msg);
    void status_callback(const uvs_message::msg::UvEmbStatus::SharedPtr msg);

private:
    void csv2vector(const string& csv_path, vector<Vector3d>& starts_, vector<Vector3d>& tasks_, vector<vector<Vector2d>>& obstacles_, int n_robot, int n_task, int n_obstacle, int ob_point);


};