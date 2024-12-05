#include "sim_map_node.hpp"

void SimServer::timer_callback()
{
	updateStates();
	message::msg::EnvState msg;
	msg.time_count= time_count_++;
	for (int i = 0; i < robot_num; i++)
	{
		message::msg::RobotState robot_state;
		robot_state.id = i;
		robot_state.pose.x = robot_states[i](0);
		robot_state.pose.y = robot_states[i](1);
		robot_state.pose.theta = robot_states[i](2);
		msg.robot_list.push_back(robot_state);
	}
	for (int i = 0; i < task_num; i++)
	{
		message::msg::TaskState task_state;
		task_state.id = i;
		task_state.pose.x = task_states[i](0);
		task_state.pose.y = task_states[i](1);
		task_state.pose.theta = task_states[i](2);
		task_state.finished = task_finished[i];
		msg.task_list.push_back(task_state);
	}
	publisher_->publish(msg);
}

void SimServer::robot_u_callback(const message::msg::RobotCtrl::SharedPtr msg)
{
	int robot_id = msg->id;
	robot_ctrls.at(robot_id) = Vector2d(msg->v, msg->w);
	robot_com_count.at(robot_id) = 0;
}

void SimServer::updateStates()
{
	//更新机器人状态
	for (int i = 0; i < robot_states.size(); i++)
	{
		if (robot_com_count[i] < perception_max)
		{
			robot_com_count[i]++;
		}
		else
		{
			break;
		}
		Vector3d state = robot_states[i];
		Vector2d ctrl = robot_ctrls[i];
		state(0) += ctrl(0) * cos(state(2)) * dt;
		state(1) += ctrl(0) * sin(state(2)) * dt;
		state(2) += ctrl(1) * dt;
		robot_states[i] = state;
	}
	//更新任务状态
	for (int i = 0; i < task_states.size(); i++)
	{
		Vector3d state = task_states[i];
		if (task_finished[i] == 0)
		{
			for (int j = 0; j < robot_states.size(); j++)
			{
				Vector3d robot_state = robot_states[j];
				if ((robot_state.head(2) - state.head(2)).norm() < finish_radius)
				{
					task_finished[i] = 1;
					break;
				}
			}
		}
	}
}

int main(int argc, char * argv[])
{
	Map_2D map;
	map.init(0.1, 0.1, 100, 100, 1, 2, 1, 4);
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

	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<SimServer>(&map));
	rclcpp::shutdown();
	return 0;
}