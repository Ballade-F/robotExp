#include "robot_node.hpp"


RobotNode::RobotNode(): Node("robot_node")
{
    //读取配置信息
    this->declare_parameter("map_dir","/home/jxl3028/Desktop/wzr/robotExp/src/config/map/map_exp");
    this->declare_parameter("robot_json_path","/home/jxl3028/Desktop/wzr/robotExp/src/config/robot/robot_0.json");
    string map_dir = this->get_parameter("map_dir").as_string();
    string robot_json_path = this->get_parameter("robot_json_path").as_string();
    string map_csv_path = map_dir + "/info.csv";
    string map_json_path = map_dir + "/batch_info.json";

    // 创建 JSON 解析器
    Json::Reader reader;
    Json::Value map_root;
    Json::Value robot_root;
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
    // 读取 robot.json 文件
    std::ifstream robot_file(robot_json_path, std::ifstream::binary);
    if (!robot_file.is_open()) 
    {
        RCLCPP_INFO(this->get_logger(), "can not open file: %s", robot_json_path.c_str());
        return ;
    }
    if (!reader.parse(robot_file, robot_root, false)) 
    {
        RCLCPP_INFO(this->get_logger(), "parse file failed: %s", robot_json_path.c_str());
        return ;
    }
    robot_file.close();
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
    //robot
    robot_id = robot_root["robot_id"].asInt();
    //planner
    planner_Ntheta = robot_root["planner_Ntheta"].asDouble();
    planner_Vmax = robot_root["planner_Vmax"].asDouble();
    planner_Wmax = robot_root["planner_Wmax"].asDouble();
    planner_Vstep = robot_root["planner_Vstep"].asInt();
    planner_Wstep = robot_root["planner_Wstep"].asInt();
    planner_TraceStep = robot_root["planner_TraceStep"].asInt();
    planner_dt = robot_root["planner_dt"].asDouble();
    planner_Rfinish = robot_root["planner_Rfinish"].asDouble();
    //mpc
    mpc_N = robot_root["mpc_N"].asInt();
    mpc_dt = robot_root["mpc_dt"].asDouble();
    mpc_Qxy = robot_root["mpc_Qxy"].asDouble();
    mpc_Qtheta = robot_root["mpc_Qtheta"].asDouble();
    mpc_Rv = robot_root["mpc_Rv"].asDouble();
    mpc_Rw = robot_root["mpc_Rw"].asDouble();
    mpc_wheel_width = robot_root["mpc_wheel_width"].asDouble();
    mpc_wheel_Vmax = robot_root["mpc_wheel_Vmax"].asDouble();
    //network
    string device_string = robot_root["device"].asString();
    string allocation_model_path = robot_root["allocation_model_path"].asString();
    string intention_model_path = robot_root["intention_model_path"].asString();

    //map
    map_p = std::make_shared<Map_2D>();
    map_p->init(map_resolution_x, map_resolution_y, map_Nx, map_Ny, map_Nrobot, map_Ntask, map_Nobstacle, map_ob_points);
	vector<Vector3d> starts(map_Nrobot, Vector3d::Zero());
    vector<Vector3d> tasks(map_Ntask, Vector3d::Zero());
    vector<vector<Vector2d>> obstacles;
    vector<Vector3d> start_(map_Nrobot, Vector3d::Zero());
    vector<Vector3d> task_(map_Ntask, Vector3d::Zero());
    csv2vector(map_csv_path, start_,task_, obstacles, map_Nrobot, map_Ntask, map_Nobstacle, map_ob_points);
    map_p->input_map(starts, tasks, obstacles);

    //hybrid_astar
    hybrid_astar_p = std::make_shared<HybridAStar>();
    hybrid_dist_astar_p = std::make_shared<HybridAStar>();
    Vector3d resolution(map_resolution_x, map_resolution_y, 2*M_PI/planner_Ntheta);
    Vector3i grid_size(map_Nx, map_Ny, planner_Ntheta);
    hybrid_astar_p->init(resolution, grid_size, map_p->grid_map, planner_Vmax, planner_Wmax, 
                             planner_Vstep, planner_Wstep,planner_TraceStep, planner_dt, planner_Rfinish, true);
    hybrid_dist_astar_p->init(resolution, grid_size, map_p->grid_map, planner_Vmax, planner_Wmax,
                                  planner_Vstep, planner_Wstep,planner_TraceStep, planner_dt, planner_Rfinish, false);
    
    //mpc
    Eigen::MatrixXd Q(3, 3);
    Q << mpc_Qxy, 0, 0,
         0, mpc_Qxy, 0,
         0, 0, mpc_Qtheta;
    Eigen::MatrixXd R(2, 2);
    R << mpc_Rv, 0,
         0, mpc_Rw;
    Eigen::MatrixXd Qf(3, 3);
    Qf << mpc_Qxy, 0, 0,
          0, mpc_Qxy, 0,
          0, 0, mpc_Qtheta;
    mpc_p = std::make_shared<MPC>();
    mpc_p->init(mpc_N, mpc_dt, mpc_wheel_Vmax, mpc_wheel_width, Q, R, Qf);
    
    //network
    
    double x_max = map_Nx * map_resolution_x;
    double y_max = map_Ny * map_resolution_y;
    double v_max_network = 0.6*planner_Vmax * ROBOT_KEYFRAME_PERIOD.count() / 1000 ;
    
    //debug 
    cout << "v_max_network: " << v_max_network  << endl;
    network_p = std::make_shared<Network>(x_max, y_max, v_max_network, map_Nrobot, map_Ntask, map_Nobstacle, map_ob_points, ROBOT_BUFFER_SIZE,
                                          allocation_model_path, intention_model_path, device_string, map_csv_path);

    robot_states.resize(map_Nrobot, Vector3d::Zero());
	task_states.resize(map_Ntask, Vector3d::Zero());
	task_finished.resize(map_Ntask, 0);
    
    //robot
    robot_p = std::make_shared<Robot>(robot_id, map_p, mpc_p, hybrid_astar_p, hybrid_dist_astar_p, network_p);

    //ros2
    publisher_ = this->create_publisher<message::msg::RobotCtrl>("robot_ctrl", 10);

    subscription_ = this->create_subscription<message::msg::EnvState>(
                    "env_state", 10, std::bind(&RobotNode::env_callback, this, _1));

    timer_ctrl = this->create_wall_timer(
                ROBOT_CONTROL_PERIOD, std::bind(&RobotNode::ctrl_timer_callback, this));
    timer_keyframe = this->create_wall_timer(
                ROBOT_KEYFRAME_PERIOD, std::bind(&RobotNode::keyframe_timer_callback, this));
    //回调组，不可重入
    cb_group_decision = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    timer_decision = this->create_wall_timer(
                ROBOT_DECISION_PERIOD, std::bind(&RobotNode::decision_timer_callback, this), cb_group_decision);
    // timer_decision = this->create_wall_timer(
    //             ROBOT_DECISION_PERIOD, std::bind(&RobotNode::decision_timer_callback, this));

} 

void RobotNode::ctrl_timer_callback()
{
    
    robot_p->pncUpdate();
    message::msg::RobotCtrl msg;
    msg.id = robot_id;
    auto ctrl_ = robot_p->ctrlOutput();
    msg.v = ctrl_(0);
    msg.w = ctrl_(1);
    publisher_->publish(msg);
    self_vw(0) = ctrl_(0);
    self_vw(1) = ctrl_(1);
}

void RobotNode::decision_timer_callback()
{
    //测回调时间
    auto start_time = std::chrono::high_resolution_clock::now();
    robot_p->decisionUpdate();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> decision_time = end_time - start_time;
    cout << "robot_id: " << robot_id << " decision_callback_time: " << decision_time.count() << endl;
    // //debug
    // RCLCPP_INFO(this->get_logger(), "robot_id: %d, target_list size: %d; state: %f, %f, %f; ctrl: %f, %f", 
    //             robot_id, target_list.size(), self_state(0), self_state(1), self_state(2), self_ctrl(0), self_ctrl(1));
}

void RobotNode::keyframe_timer_callback()
{
    robot_p->keyframeUpdate();
}

void RobotNode::env_callback(const message::msg::EnvState::SharedPtr msg)
{
    for (int i = 0; i < map_Nrobot; i++)
    {
        robot_states[i](0) = msg->robot_list[i].pose.x;
        robot_states[i](1) = msg->robot_list[i].pose.y;
        robot_states[i](2) = msg->robot_list[i].pose.theta;
    }
    for (int i = 0; i < map_Ntask; i++)
    {
        task_states[i](0) = msg->task_list[i].pose.x;
        task_states[i](1) = msg->task_list[i].pose.y;
        task_states[i](2) = msg->task_list[i].pose.theta;
        task_finished[i] = msg->task_list[i].finished;
    }
    robot_p->perceptionUpdate(robot_states, task_states, task_finished, self_vw);
}





int main(int argc, char * argv[])
{
    

    //ros2 node
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RobotNode>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    // rclcpp::spin(node);
    executor.spin();
    rclcpp::shutdown();

    return 0;
}


void RobotNode::csv2vector(const string& csv_path, vector<Vector3d>& starts_, vector<Vector3d>& tasks_, vector<vector<Vector2d>>& obstacles_, int n_robot, int n_task, int n_obstacle, int ob_point)
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
		// RCLCPP_INFO(this->get_logger(), "CONDISION:%s", (idx > n_robot+n_task && idx <= n_robot+n_task+n_obstacle*ob_point)?"TRUE":"FALSE");
        //表头
        if(idx == 0)
        {
			idx++;
            continue;
        }
        else if(idx <= n_robot)
        {
            starts_[idx-1][0] = std::stof(row[1]) * map_resolution_x * map_Nx;
            starts_[idx-1][1] = std::stof(row[2]) * map_resolution_y * map_Ny;
        }
        else if(idx <= n_robot+n_task)
        {
            tasks_[idx-n_robot-1][0] = std::stof(row[1]) * map_resolution_x * map_Nx;
            tasks_[idx-n_robot-1][1] = std::stof(row[2]) * map_resolution_y * map_Ny;
        }
        else if(idx <= n_robot+n_task+n_obstacle*ob_point)
        {
            int idx_ob = std::stoi(row[0]) - 1;
            int idx_point = (idx - n_robot - n_task - 1) - idx_ob * ob_point;
            obstacles_[idx_ob][idx_point][0] = std::stof(row[1]) * map_resolution_x * map_Nx;
            obstacles_[idx_ob][idx_point][1] = std::stof(row[2]) * map_resolution_y * map_Ny;
			// RCLCPP_INFO(this->get_logger(), "obstacle: %d, point: %d, x: %f, y: %f", idx_ob, idx_point, obstacles_[idx_ob][idx_point][0], obstacles_[idx_ob][idx_point][1]);
        }
        idx++;
    }
    csvfile.close();
}