#include "robot_node.hpp"


RobotNode::RobotNode(): Node("RobotNode")
{
    //读取配置信息
    string map_path;
    string robot_json_path;
    string map_csv_path = map_path + "/info.csv";
    string map_json_path = map_path + "/batch_info.json";

    // 创建 JSON 解析器
    Json::Reader reader;
    Json::Value map_root;
    Json::Value robot_root;
    // 读取 map.json 文件
    std::ifstream map_file(map_json_path, std::ifstream::binary);
    if (!map_file.is_open()) {
        std::cerr << "can not open file:" << map_json_path << std::endl;
        return ;
    }
    if (!reader.parse(map_file, map_root, false)) {
        std::cerr << "解析文件失败: " << map_json_path << std::endl;
        return ;
    }
    map_file.close();
    // 读取 robot.json 文件
    std::ifstream robot_file(robot_json_path, std::ifstream::binary);
    if (!robot_file.is_open()) {
        std::cerr << "无法打开文件: " << robot_json_path << std::endl;
        return ;
    }
    if (!reader.parse(robot_file, robot_root, false)) {
        std::cerr << "解析文件失败: " << robot_json_path << std::endl;
        return ;
    }
    robot_file.close();
    // 提取配置信息
    //map
    map_Nrobot = map_root["n_robot"].asInt();
    map_Ntask = map_root["n_task"].asInt();
    map_Nobstacle = map_root["n_obstacle"].asInt();
    map_ob_points = map_root["ob_point"].asInt();
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

    //map
    map_p = std::make_shared<Map_2D>();
    map_p->init(map_resolution_x, map_resolution_y, map_Nx, map_Ny, map_Nrobot, map_Ntask, map_Nobstacle, map_ob_points);
	vector<Vector3d> starts(map_Nrobot, Vector3d::Zero());
    vector<Vector3d> tasks(map_Ntask, Vector3d::Zero());
    vector<vector<Vector2d>> obstacles;
    csv2vector(map_csv_path, obstacles, map_Nrobot, map_Ntask, map_Nobstacle, map_ob_points);
    map_p->input_map(starts, tasks, obstacles);

    //hybrid_astar
    hybrid_astar_p = std::make_shared<HybridAStar>();
    hybrid_dist_astar_p = std::make_shared<HybridAStar>();
    Vector3d resolution(map_resolution_x, map_resolution_y, 2*M_PI/planner_Ntheta);
    Vector3i grid_size(map_Nx, map_Ny, planner_Ntheta);
    hybrid_astar_p->init(resolution, grid_size, map_p->grid_map, planner_Vmax, planner_Wmax, 
                             planner_Vstep, planner_Wstep, planner_dt, planner_Rfinish, true);
    hybrid_dist_astar_p->init(resolution, grid_size, map_p->grid_map, planner_Vmax, planner_Wmax,
                                  planner_Vstep, planner_Wstep, planner_dt, planner_Rfinish, false);
    
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

    message::msg::RobotCtrl msg;
    msg.id = robot_id;
    msg.v = self_ctrl(0);
    msg.w = self_ctrl(1);
    publisher_->publish(msg);
}

void RobotNode::decision_timer_callback()
{
    
    // //debug
    // RCLCPP_INFO(this->get_logger(), "robot_id: %d, target_list size: %d; state: %f, %f, %f; ctrl: %f, %f", 
    //             robot_id, target_list.size(), self_state(0), self_state(1), self_state(2), self_ctrl(0), self_ctrl(1));

}

void RobotNode::keyframe_timer_callback()
{

}

void RobotNode::env_callback(const message::msg::EnvState::SharedPtr msg)
{
    
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
    

}





int main(int argc, char * argv[])
{
    

    //ros2 node
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RobotNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}


void RobotNode::csv2vector(const string& csv_path, vector<vector<Vector2d>>& obstacles_, int n_robot, int n_task, int n_obstacle, int ob_point)
{
    obstacles_.clear();
    obstacles_.resize(n_obstacle);
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
        if(idx > n_robot+n_task && idx <= n_robot+n_task+n_obstacle*ob_point)
        {
            int idx_ob = std::stoi(row[0]) - 1;
            int idx_point = (idx - n_robot - n_task - 1) - idx_ob * ob_point;
            obstacles_[idx_ob][idx_point][0] = std::stof(row[1]);
            obstacles_[idx_ob][idx_point][1] = std::stof(row[2]);
        }
        idx++;
    }
    csvfile.close();
}