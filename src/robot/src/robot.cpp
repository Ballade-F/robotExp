#include "robot.hpp"

Robot::Robot(uint8_t robot_id_, std::shared_ptr<Map_2D> map_, std::shared_ptr<MPC> mpc_, 
             std::shared_ptr<HybridAStar> astar_, std::shared_ptr<HybridAStar> astar_dist_, std::shared_ptr<Network> network_ptr_)
        : robot_id(robot_id_), map_ptr(map_), mpc_ptr(mpc_), astar_ptr(astar_), 
          astar_dist_ptr(astar_dist_), network_ptr(network_ptr_), robot_states_keyframe(ROBOT_BUFFER_SIZE)
{
    robot_num = map_ptr->n_starts;
    task_num = map_ptr->n_tasks;
    robot_states.resize(robot_num, Vector3d::Zero());
    task_states.resize(task_num, Vector3d::Zero());
    task_finished.resize(task_num, 0);
    robot_intention.resize(robot_num, -2);//-2表示未知, -1表示无任务
    pre_allocation.resize(robot_num, -2);//-2表示不管, -1表示无任务
    target_list.reserve(task_num);
    self_ctrl = Vector2d::Zero();
}

void Robot::pncUpdate()
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
}

void Robot::perceptionUpdate(const vector<Vector3d> &robot_states_, const vector<Vector3d> &task_states_, const vector<uint8_t> &task_finished_)
{
    robot_states = robot_states_;
    task_states = task_states_;
    task_finished = task_finished_;
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

void Robot::keyframeUpdate()
{
    if (perception_counter > perception_max)
    {
        return;
    }
    robot_states_keyframe.push(robot_states);
    if(!start_flag && robot_states_keyframe.size() == ROBOT_BUFFER_SIZE)
    {
        start_flag = true;
    }
}

void Robot::decisionUpdate()
{
    if (perception_counter > perception_max)
    {
        return;
    }
    if (!start_flag && robot_states_keyframe.size()>0)
    {
        _get_allocation();
        return;
    }

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
}

const Vector2d &Robot::ctrlOutput()
{
    return self_ctrl;
}


void Robot::_get_intention(void)
{
    IntentionResult _result = network_ptr->getIntention(robot_states_keyframe, task_states, task_finished);
    robot_intention = _result.intention_id;
}

void Robot::_get_allocation(void)
{
    AllocationResult _result = network_ptr->getAllocation(robot_states, task_states, task_finished, pre_allocation);
    target_list = _result.allocation;
}