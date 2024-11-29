#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "message/msg/env_state.hpp"
#include "message/msg/robot_ctrl.hpp"
#include "message/srv/robot_intention.hpp"
#include "message/srv/robot_allocation.hpp"

#include "map.hpp"
#include "nmpc.hpp"
#include "hybrid_astar.hpp"
#include "ring_buffer.hpp"

using namespace std::chrono_literals;
using std::placeholders::_1;
//TODO: 机器人控制周期设置成可调参数
#define ROBOT_CONTROL_PERIOD (100ms)
#define ROBOT_DECISION_PERIOD (500ms)
#define ROBOT_BUFFER_SIZE 5

class RobotNode : public rclcpp::Node
{
public:
//感知
    Map_2D* map_ptr;
    uint8_t robot_id;
    int robot_num;
    int task_num;
    vector<Vector3d> robot_states;//各个机器人当前的状态 x, y, theta
    vector<Vector3d> task_states;//各个任务的位置 x, y, theta 但是theta不用
    vector<uint8_t> task_finished;//各个任务是否完成

    RingBuffer<vector<Vector3d>> robot_states_keyframe;

    bool start_flag = false;//开始标志,第一次感知到环境时设置为true

//决策
    vector<int> robot_intention;//-2表示未知, -1表示无任务
    vector<int> pre_allocation;//-2表示不管, -1表示无任务
    vector<int> target_list;
    


//规划
    HybridAStar* astar_ptr;
    HybridAStar* astar_dist_ptr;//用于决策，只需要计算距离，不需要路径，要尽可能快
    bool replan_flag = false;

//控制
    MPC* mpc_ptr;
    Vector3d self_state;
    Vector2d self_ctrl;
    bool stop_flag = true;

    RobotNode(uint8_t robot_id_, Map_2D* map_, MPC* mpc_, HybridAStar* astar_, HybridAStar* astar_dist_)
    : Node("RobotNode"), robot_id(robot_id_), map_ptr(map_), mpc_ptr(mpc_), astar_ptr(astar_), astar_dist_ptr(astar_dist_),robot_states_keyframe(ROBOT_BUFFER_SIZE)
    {
        robot_num = map_ptr->n_starts;
        task_num = map_ptr->n_tasks;
        robot_states.resize(robot_num, Vector3d::Zero());
        task_states.resize(task_num, Vector3d::Zero());
        task_finished.resize(task_num, 0);
        robot_intention.resize(robot_num, -2);//-2表示未知, -1表示无任务
        pre_allocation.resize(robot_num, -2);//-2表示不管, -1表示无任务
        target_list.reserve(task_num);
        _unfinished_tasks.reserve(task_num);
        for(int i = 0; i < robot_num; i++)
        {
            robot_states[i] = map_ptr->starts[i];
        }
        for(int i = 0; i < task_num; i++)
        {
            task_states[i] = map_ptr->tasks[i];
        }
        self_state = robot_states[robot_id];
        self_ctrl = Vector2d::Zero();

        publisher_ = this->create_publisher<message::msg::RobotCtrl>("robot_ctrl", 10);

        subscription_ = this->create_subscription<message::msg::EnvState>(
                        "env_state", 10, std::bind(&RobotNode::env_callback, this, _1));

        timer_ctrl = this->create_wall_timer(
                    ROBOT_CONTROL_PERIOD, std::bind(&RobotNode::ctrl_timer_callback, this));
        // //回调组，不可重入
        // cb_group_decision = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        // timer_decision = this->create_wall_timer(
        //             ROBOT_DECISION_PERIOD, std::bind(&RobotNode::decision_timer_callback, this), cb_group_decision);
        timer_decision = this->create_wall_timer(
                    ROBOT_DECISION_PERIOD, std::bind(&RobotNode::decision_timer_callback, this));

    } 

    


private:
    uint8_t perception_counter = 0; //感知丢失计数
    uint8_t perception_max = 5; //感知丢失最大数

    vector<int> _unfinished_tasks;


    //订阅感知信息
    rclcpp::Subscription<message::msg::EnvState>::SharedPtr subscription_;
    //发布控制信息
    rclcpp::Publisher<message::msg::RobotCtrl>::SharedPtr publisher_;
    //定时器 用于控制频率
    rclcpp::TimerBase::SharedPtr timer_ctrl;
    //定时器 用于控制频率
    rclcpp::CallbackGroup::SharedPtr cb_group_decision;
    rclcpp::TimerBase::SharedPtr timer_decision;

    void ctrl_timer_callback();
    void decision_timer_callback();
    void env_callback(const message::msg::EnvState::SharedPtr msg);

    void _get_intention(void);
    void _get_allocation(void);

};

