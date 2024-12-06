#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>

#include "network.hpp"
#include "map.hpp"
#include "nmpc.hpp"
#include "hybrid_astar.hpp"
#include "ring_buffer.hpp"

#include "rclcpp/rclcpp.hpp"



using namespace std;
using namespace Eigen;
using namespace std::chrono_literals;
using std::placeholders::_1;

#define ROBOT_BUFFER_SIZE 5

class Robot
{
    public:
//感知
    std::shared_ptr<Map_2D> map_ptr;
    uint8_t robot_id;
    int robot_num;
    int task_num;
    vector<Vector3d> robot_states;//各个机器人当前的状态 x, y, theta
    vector<Vector3d> task_states;//各个任务的位置 x, y, theta 但是theta不用
    vector<uint8_t> task_finished;//各个任务是否完成

    RingBuffer<vector<Vector3d>> robot_states_keyframe;

    bool start_flag = false;//开始标志,第一次感知到环境时设置为true

//决策
    vector<int> robot_intention;//-2表示未知, -1表示无任务
    vector<int> pre_allocation;//-2表示不管, -1表示无任务
    vector<int> target_list;
    std::shared_ptr<Network> network_ptr;


//规划
    std::shared_ptr<HybridAStar> astar_ptr;
    std::shared_ptr<HybridAStar> astar_dist_ptr;//用于决策，只需要计算距离，不需要路径，要尽可能快
    bool replan_flag = false;

//控制
    std::shared_ptr<MPC> mpc_ptr;
    Vector3d self_state;
    Vector2d self_ctrl;
    bool stop_flag = true;

    Robot(uint8_t robot_id_, std::shared_ptr<Map_2D> map_, std::shared_ptr<MPC> mpc_, 
          std::shared_ptr<HybridAStar> astar_, std::shared_ptr<HybridAStar> astar_dist_, std::shared_ptr<Network> network_ptr_);
    void pncUpdate();
    void perceptionUpdate(const vector<Vector3d>& robot_states_, const vector<Vector3d>& task_states_, 
                          const vector<uint8_t>& task_finished_);
    void keyframeUpdate();
    void decisionUpdate();
    const Vector2d& ctrlOutput();

private:
    uint8_t perception_counter = 10; //感知丢失计数,一开始不能动
    uint8_t perception_max = 5; //感知丢失最大数

    void _get_intention(void);
    void _get_allocation(void);
};