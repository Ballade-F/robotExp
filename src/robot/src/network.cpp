#include "network.hpp"

Network::Network(double x_max_, double y_max_, int n_robot_, int n_task_, int n_obstacle_, int ob_point_, int r_point_,
            string allocation_model_path_, string intention_model_path_, string device_string_, string map_csv_path_)
{
    x_max = x_max_;
    y_max = y_max_;
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
        allocation_model.run_method("config_script",n_robot, n_task, n_obstacle, batch_size, ob_point, device_string);
    }
    else
    {
        cerr << "No config_script found in the allocation model\n";
    }

    if (intention_model.find_method("config_script"))
    {
        intention_model.run_method("config_script",n_robot, n_task, n_obstacle, batch_size, ob_point);
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
    ifstream csvfile(map_csv_path_);
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
            obstacles[0][idx_ob][idx_point][0] = std::stof(row[1]);
            obstacles[0][idx_ob][idx_point][1] = std::stof(row[2]);
        }
        idx++;
    }
    csvfile.close();

}

const AllocationResult& Network::getAllocation(const vector<Vector3d> &robot_states_, const vector<Vector3d> &task_states_, vector<uint8_t> task_finished_,  vector<int> pre_allocation_)
{
    //寻找有效的robot
    int valid_robot_num = 0;
    vector<int> valid_robot;
    for (int i = 0; i < n_robot; i++)
    {
        // -2表示未分配, -1表示停止，>=0表示任务
        if (pre_allocation_[i] == -2)
        {
            valid_robot.push_back(i);
            valid_robot_num++;
        }
        else if (pre_allocation_[i] >= 0)
        {
            valid_robot.push_back(i);
            valid_robot_num++;
            task_finished_[pre_allocation_[i]] = 1;//标记任务已经被分配
        }
    }
    torch::Tensor robot_tensor = torch::zeros({batch_size, valid_robot_num, 3}, torch::kFloat32);
    for (int i = 0; i < valid_robot_num; i++)
    {
        //如果是未分配，就是机器人的位置，如果是分配了，就是任务的位置
        if(pre_allocation_[valid_robot[i]] == -2)
        {
            robot_tensor[0][i][0] = robot_states_[valid_robot[i]][0] / x_max;
            robot_tensor[0][i][1] = robot_states_[valid_robot[i]][1] / y_max;
            robot_tensor[0][i][2] = -1;//-1表示机器人，0表示任务
        }
        else
        {
            robot_tensor[0][i][0] = task_states_[pre_allocation_[valid_robot[i]]][0] / x_max;
            robot_tensor[0][i][1] = task_states_[pre_allocation_[valid_robot[i]]][1] / y_max;
            robot_tensor[0][i][2] = -1;
        }
    }

    //寻找有效的task
    int valid_task_num = 0;
    vector<int> valid_task;
    for (int i = 0; i < n_task; i++)
    {
        if (task_finished_[i] == 0)
        {
            valid_task.push_back(i);
            valid_task_num++;
        }
    }
    torch::Tensor task_tensor = torch::zeros({batch_size, valid_task_num, 3}, torch::kFloat32);
    for (int i = 0; i < valid_task_num; i++)
    {
        task_tensor[0][i][0] = task_states_[valid_task[i]][0] / x_max;
        task_tensor[0][i][1] = task_states_[valid_task[i]][1] / y_max;
        task_tensor[0][i][2] = 0;
    }

    //forward
    // //debug
    cout << robot_tensor << endl;
    cout << task_tensor << endl;
    cout << obstacles << endl;
    cout << valid_robot_num << " " << valid_task_num << endl;
    allocation_model.run_method("config_script",valid_robot_num, valid_task_num, n_obstacle, batch_size, ob_point, device_string);
    allocation_inputs.clear();
    allocation_inputs.push_back(robot_tensor.to(device));
    allocation_inputs.push_back(task_tensor.to(device));
    allocation_inputs.push_back(obstacles.to(device));
    allocation_inputs.push_back(torch::randn({1, 2, 2}).to(device));//costmap，没用

    auto start_time = std::chrono::high_resolution_clock::now();
    auto output = allocation_model.forward(allocation_inputs);//3个tensor的元组，只用第一个，维度(batch, seq)
    auto end_time = std::chrono::high_resolution_clock::now();
    double allocation_time = (end_time - start_time).count();

    //取出output的第一个tensor
    auto output_tensor = output.toTuple()->elements()[0].toTensor();//(batch, seq)
    allocation_result.allocation.clear();
    for (int i = 0; i < output_tensor.size(1); i++)
    {
        int _id = output_tensor[0][i].item<int>();
        if(_id == 0)
        {
            break;
        }
        else
        {
            int valid_task_id = _id - valid_robot_num;
            int task_id = valid_task[valid_task_id];
            allocation_result.allocation.push_back(task_id);
        }
    }
    allocation_result.allocation_time = allocation_time;

    return allocation_result;

}

// x_r: (batch, n_robot,r_points, 2), x_t: (batch, self.n_task, 3), x_ob: (batch, n_obstacle, ob_points, 2)
// robot_states_keyframe_: r_points, n_robot, 3
const IntentionResult& Network::getIntention(const RingBuffer<vector<Vector3d>> &robot_states_keyframe_, const vector<Vector3d> &task_states_, const vector<uint8_t> &task_finished_)
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
    intention_inputs.push_back(torch::tensor(false).to(device));

    auto start_time = std::chrono::high_resolution_clock::now();
    auto output = intention_model.forward(intention_inputs).toTensor();//(batch, n_robot, task_net_num)
    auto end_time = std::chrono::high_resolution_clock::now();
    double intention_time = (end_time - start_time).count();

    //softmax and get the max id
    auto output_softmax = torch::softmax(output, 2);//(batch, n_robot, task_net_num)
    auto output_max = output_softmax.max(2);//(batch, n_robot)， max返回的是值和索引
    auto output_max_id = std::get<1>(output_max);//(batch, n_robot)
    auto output_max_prob = std::get<0>(output_max);//(batch, n_robot)

    //output
    intention_result.intention_id.resize(n_robot);
    intention_result.intention_prob.resize(n_robot);
    for (int i = 0; i < n_robot; i++)
    {
        intention_result.intention_id[i] = output_max_id[0][i].item<int>();
        if(intention_result.intention_id[i] == n_task)
        {
            intention_result.intention_id[i] = -1;
        }
        intention_result.intention_prob[i] = output_max_prob[0][i].item<double>();
    }
    intention_result.intention_time = intention_time;

    return intention_result;
}


