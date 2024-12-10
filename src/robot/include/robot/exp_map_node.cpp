#include "exp_map_node.hpp"

ExpServer::ExpServer(): Node("exp_node")
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
	vector<Vector3d> start_(map_Nrobot, Vector3d::Zero());
    vector<Vector3d> task_(map_Ntask, Vector3d::Zero());
    map_ptr->input_map(start_, task_, obstacles);

}

void ExpServer::csv2vector(const string& csv_path, vector<vector<Vector2d>>& obstacles_, int n_robot, int n_task, int n_obstacle, int ob_point)
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