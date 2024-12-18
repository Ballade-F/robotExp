#include "sim_map_node.hpp"

void SimServer::timer_callback()
{
	updateStates();
	message::msg::EnvState msg;
	msg.time_count= time_count_++;
	for (int i = 0; i < map_Nrobot; i++)
	{
		message::msg::RobotState robot_state;
		robot_state.id = i;
		robot_state.pose.x = robot_states[i](0);
		robot_state.pose.y = robot_states[i](1);
		robot_state.pose.theta = robot_states[i](2);
		msg.robot_list.push_back(robot_state);
	}
	for (int i = 0; i < map_Ntask; i++)
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

SimServer::SimServer(const vector<Vector3d> &start_, const vector<Vector3d> &task_): Node("sim_node")
{
	//读取配置信息
    this->declare_parameter("map_dir","/home/jxl3028/Desktop/wzr/robotExp/src/config/map/map_0");
	string map_dir = this->get_parameter("map_dir").as_string();
	string map_csv_path = map_dir + "/info.csv";
    string map_json_path = map_dir + "/batch_info.json";

	// 创建 JSON 解析器
	Json::Reader reader;
	Json::Value map_root;
	// 读取 map.json 文件
	std::ifstream map_file(map_json_path, std::ifstream::binary);
	if (!map_file.is_open()) 
	{
		RCLCPP_INFO(this->get_logger(), "can not open file: %s", map_json_path.c_str());
		return ;
	}
	if (!reader.parse(map_file, map_root, false)) 
	{
		RCLCPP_INFO(this->get_logger(), "parse file failed: %s", map_json_path.c_str());
		return ;
	}
	map_file.close();
	// 提取配置信息
	//map
	map_Nrobot = map_root["n_robot"].asInt();
	map_Ntask = map_root["n_task"].asInt();
	map_Nobstacle = map_root["n_obstacle"].asInt();
	map_ob_points = map_root["ob_points"].asInt();
	map_Nx = map_root["n_x"].asInt();
	map_Ny = map_root["n_y"].asInt();
	map_resolution_x = map_root["resolution_x"].asDouble();
	map_resolution_y = map_root["resolution_y"].asDouble();

	vector<vector<Vector2d>> obstacles;
	map_ptr = std::make_shared<Map_2D>();
	map_ptr->init(map_resolution_x, map_resolution_y, map_Nx, map_Ny, map_Nrobot, map_Ntask, map_Nobstacle, map_ob_points);
	csv2vector(map_csv_path, obstacles, map_Nrobot, map_Ntask, map_Nobstacle, map_ob_points);
	map_ptr->input_map(start_, task_, obstacles);

	robot_states.resize(map_Nrobot, Vector3d::Zero());
	robot_ctrls.resize(map_Nrobot, Vector2d::Zero());
	task_states.resize(map_Ntask, Vector3d::Zero());
	task_finished.resize(map_Ntask, 0);
	for(int i = 0; i < map_Nrobot; i++)
	{
		robot_states[i] = map_ptr->starts[i];
	}
	for(int i = 0; i < map_Ntask; i++)
	{
		task_states[i] = map_ptr->tasks[i];
	}
	time_count_ = 0;

	robot_com_count.resize(map_Nrobot, 0);



	//rviz2
	occupancy_grid_publisher_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("occupancy_grid", 10);
	occupancy_grid_msg = std::make_shared<nav_msgs::msg::OccupancyGrid>();
	// 设置头信息
	// occupancy_grid_msg->header.stamp = this->now();
	occupancy_grid_msg->header.frame_id = "map";
	// 设置地图信息
	occupancy_grid_msg->info.resolution = map_resolution_x; // 假设 x 和 y 分辨率相同
	occupancy_grid_msg->info.width = map_Nx;
	occupancy_grid_msg->info.height = map_Ny;
	occupancy_grid_msg->info.origin.position.x = 0.0;
	occupancy_grid_msg->info.origin.position.y = 0.0;
	occupancy_grid_msg->info.origin.position.z = 0.0;
	occupancy_grid_msg->info.origin.orientation.w = 1.0;
	// 将 map_ptr 中的 grid_map 数据填充到 occupancy_grid_msg 中
	occupancy_grid_msg->data.resize(map_Nx * map_Ny);
	for (int i = 0; i < map_Nx; ++i)
	{
		for (int j = 0; j < map_Ny; ++j)
		{
			occupancy_grid_msg->data[j*map_Nx + i ] = static_cast<int8_t>(map_ptr->grid_map[j*map_Nx + i])*100;
		}
	}

	marker_array_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("robot_task_states", 10);
	marker_array_msg = std::make_shared<visualization_msgs::msg::MarkerArray>();

	
	publishOccupancyGrid();
	publishRobotTaskStates();

	publisher_ = this->create_publisher<message::msg::EnvState>("env_state", 10);
	subscription_ = this->create_subscription<message::msg::RobotCtrl>(
					"robot_ctrl", 10, std::bind(&SimServer::robot_u_callback, this, _1));
	timer_ = this->create_wall_timer(
				100ms, std::bind(&SimServer::timer_callback, this));
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
			continue;
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
	//rviz2
	publishOccupancyGrid();
	publishRobotTaskStates();
}

int main(int argc, char * argv[])
{
	vector<Vector3d> starts;
    starts.push_back(Vector3d(1.0, 1.0, 0.0));
	starts.push_back(Vector3d(1.0, 9.0, 0.0));
    vector<Vector3d> tasks;
	tasks.push_back(Vector3d(5.0, 3.0, 0.0));
	tasks.push_back(Vector3d(9.0, 1.0, 0.0));
    tasks.push_back(Vector3d(9.0, 9.0, 0.0));

	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<SimServer>(starts, tasks));
	rclcpp::shutdown();
	return 0;
}

void SimServer::csv2vector(const string& csv_path, vector<vector<Vector2d>>& obstacles_, int n_robot, int n_task, int n_obstacle, int ob_point)
{
    obstacles_.clear();
    for(int i = 0; i < n_obstacle; i++)
    {
        vector<Vector2d> obstacle(ob_point, Vector2d::Zero());
        obstacles_.push_back(obstacle);
    }
    ifstream csvfile(csv_path);
    std::string line;
    int idx = 0;
    while (std::getline(csvfile, line)) 
    {
        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> row;
        while (std::getline(ss, token, ',')) 
        {
            row.push_back(token);
        }
		RCLCPP_INFO(this->get_logger(), "CONDISION:%s", (idx > n_robot+n_task && idx <= n_robot+n_task+n_obstacle*ob_point)?"TRUE":"FALSE");
        if(idx > n_robot+n_task && idx <= n_robot+n_task+n_obstacle*ob_point)
        {
            int idx_ob = std::stoi(row[0]) - 1;
            int idx_point = (idx - n_robot - n_task - 1) - idx_ob * ob_point;
            obstacles_[idx_ob][idx_point][0] = std::stof(row[1]) * map_resolution_x * map_Nx;
            obstacles_[idx_ob][idx_point][1] = std::stof(row[2]) * map_resolution_y * map_Ny;
			RCLCPP_INFO(this->get_logger(), "obstacle: %d, point: %d, x: %f, y: %f", idx_ob, idx_point, obstacles_[idx_ob][idx_point][0], obstacles_[idx_ob][idx_point][1]);
        }
        idx++;
    }
    csvfile.close();
}

void SimServer::publishOccupancyGrid()
{
	// 发布消息
	occupancy_grid_msg->header.stamp = this->now();
	occupancy_grid_publisher_->publish(*occupancy_grid_msg);
}

void SimServer::publishRobotTaskStates()
{
	marker_array_msg->markers.clear();
	for (size_t i = 0; i < robot_states.size(); ++i)
	{
		visualization_msgs::msg::Marker marker;
		marker.header.stamp = this->now();
		marker.header.frame_id = "map";
		marker.ns = "robots";
		marker.id = i;
		marker.type = visualization_msgs::msg::Marker::ARROW;
		marker.action = visualization_msgs::msg::Marker::ADD;
		marker.pose.position.x = robot_states[i][0];
		marker.pose.position.y = robot_states[i][1];
		marker.pose.position.z = 0.0;
		// 使用四元数设置 orientation
        tf2::Quaternion quat;
        quat.setRPY(0, 0, robot_states[i][2]);
        marker.pose.orientation.x = quat.x();
        marker.pose.orientation.y = quat.y();
        marker.pose.orientation.z = quat.z();
        marker.pose.orientation.w = quat.w();
		marker.scale.x = 0.2;
		marker.scale.y = 0.2;
		marker.scale.z = 0.2;
		marker.color.a = 1.0;
		marker.color.r = 0.0;
		marker.color.g = 1.0;
		marker.color.b = 0.0;

		marker_array_msg->markers.push_back(marker);
	}
	for (size_t i = 0; i < task_states.size(); ++i)
	{
		visualization_msgs::msg::Marker marker;
		marker.header.stamp = this->now();
		marker.header.frame_id = "map";
		marker.ns = "tasks";
		marker.id = i;
		marker.type = visualization_msgs::msg::Marker::SPHERE;
		marker.action = visualization_msgs::msg::Marker::ADD;
		marker.pose.position.x = task_states[i][0];
		marker.pose.position.y = task_states[i][1];
		marker.pose.position.z = 0.0;
		marker.pose.orientation.x = 0.0;
		marker.pose.orientation.y = 0.0;
		marker.pose.orientation.z = 0.0;
		marker.pose.orientation.w = 1.0;
		marker.scale.x = 0.2;
		marker.scale.y = 0.2;
		marker.scale.z = 0.2;
		marker.color.a = 1.0;
		if(task_finished[i] == 1)
		{
			marker.color.r = 0.0;
			marker.color.g = 1.0;
			marker.color.b = 0.0;
		}
		else
		{
			marker.color.r = 0.0;
			marker.color.g = 0.0;
			marker.color.b = 1.0;
		}
		marker_array_msg->markers.push_back(marker);
	}
	marker_array_publisher_->publish(*marker_array_msg);
}
