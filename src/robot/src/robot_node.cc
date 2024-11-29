#include "robot_node.hpp"

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
    // robot_states_keyframe.push(robot_states);

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

void RobotNode::_wait_service(void)
{
//等待服务端上线
    while (!client_intention->wait_for_service(std::chrono::seconds(1)))
    {
        //等待时检测rclcpp的状态
        if (!rclcpp::ok())
        {
            RCLCPP_ERROR(this->get_logger(), "client_intention interrupted while waiting for service. Exiting.");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "waiting for intention service to appear...");
    }
    RCLCPP_INFO(this->get_logger(), "intention service online");
    while(!client_allocation->wait_for_service(std::chrono::seconds(1)))
    {
        if (!rclcpp::ok())
        {
            RCLCPP_ERROR(this->get_logger(), "client_allocation interrupted while waiting for service. Exiting.");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "waiting for allocation service to appear...");
    }
    RCLCPP_INFO(this->get_logger(), "allocation service online");
}

void RobotNode::_get_intention(void)
{
    message::srv::RobotIntention::Request::SharedPtr request = std::make_shared<message::srv::RobotIntention::Request>();
    auto result = client_intention->async_send_request(request);
    //wait for result
    std::future_status status = result.wait_for(5s); 
    if (status == std::future_status::ready) 
    {
        for(int i = 0; i < robot_num; i++)
        {
            robot_intention[i] = result.get()->intention_result.vec_data[i];
        }
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to get intention");
    }
}

void RobotNode::_get_allocation(void)
{
    message::srv::RobotAllocation::Request::SharedPtr request = std::make_shared<message::srv::RobotAllocation::Request>();
    request->algorithm = 2; //0:greedy, 1:ga, 2:network
    for(int i = 0; i < robot_num; i++)
    {
        request->pre_allocation.vec_data[i] = pre_allocation[i];
    }
    auto result = client_allocation->async_send_request(request);
    //wait for result
    std::future_status status = result.wait_for(5s); 
    if (status == std::future_status::ready) 
    {
        target_list.clear();
        for(int i = 0; i < result.get()->allocation_result.vec_data.size(); i++)
        {
            target_list.push_back(result.get()->allocation_result.vec_data[i]);
        }
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to get allocation");
    }
}

void RobotNode::_reallocation(void)
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