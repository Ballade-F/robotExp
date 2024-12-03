#include "robot_node.hpp"

RobotNode::RobotNode(uint8_t robot_id_, Map_2D* map_, MPC* mpc_, HybridAStar* astar_, HybridAStar* astar_dist_)
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


void RobotNode::ctrl_timer_callback()
{
    if (!start_flag)
    {
        return;
    }
    if (perception_counter > perception_max)
    {
        stop_flag = true;
    }
    perception_counter++;

    if(target_list.size() == 0)
    {
        replan_flag = false;
        stop_flag = true;
    }

    if (replan_flag)
    {
        PlanResult& plan_result = astar_ptr->plan(self_state, task_states[target_list[0]]);
        if (plan_result.success)
        {
            int trace_num = plan_result.trace.size();
            VectorXd x_ref(trace_num), y_ref(trace_num), theta_ref(trace_num), v_ref(trace_num), w_ref(trace_num);
            for (int i = 0; i < trace_num; i++)
            {
                x_ref(i) = plan_result.trace[i](0);
                y_ref(i) = plan_result.trace[i](1);
                theta_ref(i) = plan_result.trace[i](2);
                v_ref(i) = plan_result.trace_controls[i](0);
                w_ref(i) = plan_result.trace_controls[i](1);
            }
            mpc_ptr->setTrackReference(x_ref, y_ref, theta_ref, v_ref, w_ref);
            replan_flag = false;
            stop_flag = false;
        }
        else
        {
            stop_flag = true;
        }
        
    }

    if(!stop_flag)
    {
        VectorXd control;
        VectorXd state(5);
        state << self_state(0), self_state(1), self_state(2), self_ctrl(0), self_ctrl(1);
        bool done = mpc_ptr->update(state, control);
        if (done)
        {
            stop_flag = true;
            control = VectorXd::Zero(2);
        }
        self_ctrl(0) = control(0);
        self_ctrl(1) = control(1); 
    }
    else
    {
        self_ctrl = Vector2d::Zero();
    }
    message::msg::RobotCtrl msg;
    msg.id = robot_id;
    msg.v = self_ctrl(0);
    msg.w = self_ctrl(1);
    publisher_->publish(msg);
}

void RobotNode::decision_timer_callback()
{
    if (!start_flag)
    {
        return;
    }
    robot_states_keyframe.push(robot_states);

    //intention
    _get_intention();
    //如果检测到机器人停车，修改pre_allocation给决策用
    for(int i = 0; i < robot_num; i++)
    {
        if(robot_intention[i] == -1)
        {
            pre_allocation[i] = -1;
        }
        else
        {
            pre_allocation[i] = -2;
        }
    }
    //allocation
    bool reallocation_flag = false;
    //任务列表为空或者任务列表中有任务被完成
    if (target_list.size() == 0)
    {
        reallocation_flag = true;
    }
    else
    {
        for (int i = 0; i < target_list.size(); i++)
        {
            if (task_finished[target_list[i]] == 1)
            {
                reallocation_flag = true;
                break;
            }
        }
    }
    if (reallocation_flag)
    {
        _get_allocation();
        if (target_list.size() == 0)
        {
            stop_flag = true;
            replan_flag = false;
            return;
        }
    }

    //冲突消解，最多robot_num-1次
    bool conflict_flag = false;
    int _task_id = -1;
    int _robot_conflict_id = -1;
    for(int _iterations = 0; _iterations < robot_num-1; _iterations++)
    {
        conflict_flag = false;
        _task_id = -1;
        _robot_conflict_id = -1;
        for (int i = 0; i < robot_num; i++)
        {
            if(i==robot_id)
            {
                continue;
            }
            //意图相同
            if(robot_intention[i] == target_list[0])
            {
                _task_id = target_list[0];
                //计算各自的代价
                PlanResult& plan_result_other = astar_dist_ptr->plan(robot_states[i], task_states[_task_id]);
                PlanResult& plan_result_self = astar_dist_ptr->plan(self_state, task_states[_task_id]);
                if(plan_result_other.success && plan_result_self.success)
                {
                    double cost_other = plan_result_other.cost;
                    double cost_self = plan_result_self.cost;
                    if(cost_other < cost_self)
                    {
                        conflict_flag = true;
                        _robot_conflict_id = i;
                        break;
                    }
                }
                else if(plan_result_other.success)
                {
                    conflict_flag = true;
                    _robot_conflict_id = i;
                    break;
                }

            }
        }
        //没有冲突
        if(!conflict_flag)
        {
            break;
        }
        //冲突消解
        else
        {
            reallocation_flag = true;
            pre_allocation[_robot_conflict_id] = _task_id;//让出任务
            _get_allocation();
            if (target_list.size() == 0)
            {
                stop_flag = true;
                replan_flag = false;
                return;
            }
        }
    }
    if (reallocation_flag)
    {
        replan_flag = true;
    }
    //debug
    RCLCPP_INFO(this->get_logger(), "robot_id: %d, target_list size: %d; state: %f, %f, %f; ctrl: %f, %f", 
                robot_id, target_list.size(), self_state(0), self_state(1), self_state(2), self_ctrl(0), self_ctrl(1));

}

void RobotNode::env_callback(const message::msg::EnvState::SharedPtr msg)
{
    start_flag = true;
    for (int i = 0; i < robot_num; i++)
    {
        robot_states[i](0) = msg->robot_list[i].pose.x;
        robot_states[i](1) = msg->robot_list[i].pose.y;
        robot_states[i](2) = msg->robot_list[i].pose.theta;
    }
    for (int i = 0; i < task_num; i++)
    {
        task_states[i](0) = msg->task_list[i].pose.x;
        task_states[i](1) = msg->task_list[i].pose.y;
        task_states[i](2) = msg->task_list[i].pose.theta;
        task_finished[i] = msg->task_list[i].finished;
    }
    self_state = robot_states[robot_id];
    perception_counter = 0;//感知到了

    //更新任务列表
    while(target_list.size() >0)
    {
        int target_id = target_list[0];
        if(task_finished[target_id] == 1)
        {
            target_list.erase(target_list.begin());
            replan_flag = true;
        }
        else
        {
            break;
        }
    }
    if (target_list.size() == 0)
    {
        replan_flag = false;
        stop_flag = true;
    }

}



void RobotNode::_get_intention(void)
{

}

void RobotNode::_get_allocation(void)
{

}



int main(int argc, char * argv[])
{
    //map
    Map_2D map(0.1, 0.1, 100, 100, 1, 2, 1, 4);
	vector<Vector3d> starts;
    starts.push_back(Vector3d(0.5, 0.5, 0.0));
    vector<Vector3d> tasks;
	tasks.push_back(Vector3d(8.0, 2.0, 0.0));
    tasks.push_back(Vector3d(9.5, 9.5, 0.0));
    vector<vector<Vector2d>> obstacles;
    vector<Vector2d> obstacle;
    obstacle.push_back(Vector2d(3.0, 3.0));
    obstacle.push_back(Vector2d(3.0, 5.0));
    obstacle.push_back(Vector2d(6.0, 5.0));
    obstacle.push_back(Vector2d(6.0, 3.0));
    obstacles.push_back(obstacle);
    map.input_map(starts, tasks, obstacles);

    //hybrid_astar
    Vector3d resolution(0.1, 0.1, M_PI/8);
    Vector3i grid_size(100, 100, 16);
    double max_v = 0.3;
    double max_w = M_PI/4;
    int step_v = 2;
    int step_w = 2;
    double dt = 0.5;
    double finish_radius = 0.3;
    HybridAStar hybrid_astar(resolution, grid_size, map.grid_map, max_v, max_w, step_v, step_w, dt, finish_radius);
    HybridAStar hybrid_dist_astar(resolution, grid_size, map.grid_map, max_v, max_w, step_v, step_w, dt, finish_radius, false);
    //mpc
    int N = 10;
    double dT = 0.1;
    Eigen::MatrixXd Q(3, 3);
    Q << 10, 0, 0,
         0, 10, 0,
         0, 0, 1;
    Eigen::MatrixXd R(2, 2);
    R << 1, 0,
         0, 0.1;
    Eigen::MatrixXd Qf(3, 3);
    Qf << 10, 0, 0,
          0, 10, 0,
          0, 0, 1;
    MPC mpc(N, dT, 0.4, 0.238, Q, R, Qf);

    //ros2 node
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RobotNode>(0, &map, &mpc, &hybrid_astar, &hybrid_dist_astar);
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}