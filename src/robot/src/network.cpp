#include "network.hpp"

Network::Network(double x_max_, double y_max, int n_robot_, int n_task_, int n_obstacle_, int ob_point_, int r_point_,
            string allocation_model_path_, string intention_model_path_, string device_string_, string map_path_)
{
    n_robot = n_robot_;
    n_task = n_task_;
    n_obstacle = n_obstacle_;
    ob_point = ob_point_;
    r_point = r_point_;
    allocation_model_path = allocation_model_path_;
    intention_model_path = intention_model_path_;
    device_string = device_string_;

    //如果是cuda:id，分离出id
    if (device_string.find("cuda:") != string::npos)
    {
        device = torch::Device(torch::kCUDA, stoi(device_string.substr(5)));
        //debug
        cout << "Training on GPU " << stoi(device_string.substr(5)) << endl;
    }
    else
    {
        device = torch::Device(device_string == "cuda" ? torch::kCUDA : torch::kCPU);
        //debug
        cout << "Training on " << (device_string == "cuda" ? "GPU" : "CPU") << endl;
    }

    // Load the model
    try
    {
        allocation_model = torch::jit::load(allocation_model_path);
        allocation_model.to(device);
        allocation_model.eval();
        cout << "Allocation model loaded successfully!" << endl;
    }
    catch (const c10::Error &e)
    {
        cerr << "Error loading the allocation model\n";
    }

    try
    {
        intention_model = torch::jit::load(intention_model_path);
        intention_model.to(device);
        intention_model.eval();
        cout << "Intention model loaded successfully!" << endl;
    }
    catch (const c10::Error &e)
    {
        cerr << "Error loading the intention model\n";
    }

    //config
    if (allocation_model.find_method("config_script"))
    {
        allocation_model.run_method("config_script",n_robot, n_task, n_obstacle, ob_point, batch_size, device_string);
    }
    else
    {
        cerr << "No config_script found in the allocation model\n";
    }

    if (intention_model.find_method("config_script"))
    {
        intention_model.run_method("config_script",n_robot, n_task, n_obstacle, ob_point, batch_size);
    }
    else
    {
        cerr << "No config_script found in the intention model\n";
    }

    //obstacles
    // obstacles.clear();
    // for(int i = 0; i < n_obstacle; i++)
    // {
    //     vector<Vector2d> obstacle;
    //     obstacle.resize(ob_point, Vector2d::Zero());
    //     obstacles.push_back(obstacle);
    // }

    obstacles = torch::zeros({batch_size,n_obstacle, ob_point, 2}, torch::kFloat32);
    // Load the obstacles, map_path_文件夹下的info.csv文件
    ifstream csvfile(map_path_ + "/info.csv");
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
        if(idx > n_robot+n_task)
        {
            int idx_ob = std::stoi(row[0]) - 1;
            int idx_point = (idx - n_robot - n_task - 1) - idx_ob * ob_point;
            obstacles[0][idx_ob][idx_point][0] = std::stof(row[1]);
            obstacles[0][idx_ob][idx_point][1] = std::stof(row[2]);
        }
        idx++;
    }

}

//x_r: (batch, n_robot,r_points, 2), x_t: (batch, self.n_task, 3), x_ob: (batch, n_obstacle, ob_points, 2)
//robot_states_keyframe_: r_points, n_robot, 3
vector<int> Network::getIntention(const RingBuffer<vector<Vector3d>> &robot_states_keyframe_, const vector<Vector3d> &task_states_, const vector<uint8_t> &task_finished_)
{
     //robot tensor 归一化
    torch::Tensor robot_tensor = torch::zeros({batch_size, n_robot, r_point, 2}, torch::kFloat32);
    for (int i = 0; i < n_robot; i++)
    {
        for (int j = 0; j < r_point; j++)
        {
            //robot_tensor j=0对应最老的点，j=r_point-1对应最新的点，robot_states_keyframe_则相反
            robot_tensor[0][i][j][0] = robot_states_keyframe_[r_point-1-j][i][0] / x_max;
            robot_tensor[0][i][j][1] = robot_states_keyframe_[r_point-1-j][i][1] / y_max;
        }
    }

    //task tensor
    int task_net_num = n_task + 1;//最后一个是虚拟任务，表示停止，为-1, -1, 0
    torch::Tensor task_tensor = torch::zeros({batch_size, task_net_num, 3}, torch::kFloat32);
    for (int i = 0; i < n_task; i++)
    {
        task_tensor[0][i][0] = task_states_[i][0] / x_max;
        task_tensor[0][i][1] = task_states_[i][1] / y_max;
        task_tensor[0][i][2] = task_finished_[i];
    }
    task_tensor[0][n_task][0] = -1;
    task_tensor[0][n_task][1] = -1;
    task_tensor[0][n_task][2] = 0;

    //forward
    intention_inputs.clear();
    intention_inputs.push_back(robot_tensor.to(device));
    intention_inputs.push_back(task_tensor.to(device));
    intention_inputs.push_back(obstacles.to(device));
    // intention_inputs.push_back(torch::tensor(false).to(device));

    auto start_time = std::chrono::high_resolution_clock::now();
    auto output = intention_model.forward(intention_inputs).toTensor();
    auto end_time = std::chrono::high_resolution_clock::now();
    intention_time = (end_time - start_time).count();

    //softmax and get the max id
    auto output_cpu = output.to(torch::kCPU);
    auto output_a = output_cpu.accessor<float, 3>();
    vector<int> intention(n_robot, -2);
    for (int i = 0; i < n_robot; i++)
    {
        float max_p = 0;
        int max_id = -2;
        for (int j = 0; j < task_net_num; j++)
        {
            if (output_a[0][i][j] > max_p)
            {
                max_p = output_a[0][i][j];
                max_id = j - 1;
            }
        }
        intention[i] = max_id;
    }

    //output
    


    
}


