#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "map.hpp"
#include "message/msg/env_state.hpp"
#include "message/msg/robot_ctrl.hpp"

using namespace std::chrono_literals;
using std::placeholders::_1;

class SimServer : public rclcpp::Node
{
public:
    SimServer(Map_2D* map_)
    : Node("SimServer"), map_ptr(map_)
    {
        robot_num = map_ptr->n_starts;
        task_num = map_ptr->n_tasks;
        robot_states.resize(robot_num, Vector3d::Zero());
        robot_ctrls.resize(robot_num, Vector2d::Zero());
        task_states.resize(task_num, Vector3d::Zero());
        task_finished.resize(task_num, 0);
        for(int i = 0; i < robot_num; i++)
        {
            robot_states[i] = map_ptr->starts[i];
        }
        for(int i = 0; i < task_num; i++)
        {
            task_states[i] = map_ptr->tasks[i];
        }
        time_count_ = 0;

        robot_com_count.resize(robot_num, 0);

        publisher_ = this->create_publisher<message::msg::EnvState>("env_state", 10);
        subscription_ = this->create_subscription<message::msg::RobotCtrl>(
                        "robot_ctrl", 10, std::bind(&SimServer::robot_u_callback, this, _1));
        timer_ = this->create_wall_timer(
                    100ms, std::bind(&SimServer::timer_callback, this));
    }

    int robot_num;
    int task_num;
    vector<Vector3d> robot_states;//各个机器人当前的状态 x, y, theta
    vector<Vector2d> robot_ctrls;//各个机器人当前的控制 v, w
    vector<Vector3d> task_states;//各个任务的位置 x, y, theta 但是theta不用
    vector<uint8_t> task_finished;//各个任务是否完成
    vector<uint8_t> robot_com_count;//机器人通信计数器
    uint8_t perception_max = 5; //感知丢失最大数
    void updateStates();

private:
    Map_2D* map_ptr;
    double dt = 0.1;
    double finish_radius = 0.3;

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
    
};