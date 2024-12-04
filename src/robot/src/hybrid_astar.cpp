#include "hybrid_astar.hpp"

void HybridAStar::init(Vector3d resolution, Vector3i grid_size, uint8_t *grid_map, double max_v, double max_w, int step_v, int step_w, int trace_step_, double dt, double finish_radius, bool path_flag_)
{
    this->resolution = resolution;
    this->grid_size = grid_size;
    this->grid_map = grid_map;
    this->finish_radius = finish_radius;
    this->max_v = max_v;
    this->max_w = max_w;
    this->step_v = step_v;
    this->step_w = step_w;
    this->dt = dt;
    this->path_flag = path_flag_;
    this->trace_step = trace_step_;
    map_max = Vector3d(grid_size[0]*resolution[0],grid_size[1]*resolution[1],2*M_PI);
    nodes.resize(grid_size[0]*grid_size[1]*grid_size[2],nullptr);
    for (int i = 0; i < grid_size[0]; i++)
    {
        for (int j = 0; j < grid_size[1]; j++)
        {
            for (int k = 0; k < grid_size[2]; k++)
            {
                Node3D* node = new Node3D;
                node->index = Vector3i(i,j,k);
                node->state = Vector3d(i*resolution[0],j*resolution[1],k*resolution[2]);
                nodes[i*grid_size[1]*grid_size[2]+j*grid_size[2]+k] = node;
            }
        }
    }
    _generateNeighborList();
}

PlanResult &HybridAStar::plan(Vector3d start, Vector3d goal)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    reset();
    int start_index = getNodeIndex(start);
    int goal_index = getNodeIndex(goal);
    Node3D *start_node = nodes[start_index];
    Node3D *goal_node = nodes[goal_index];
    start_node->g = 0;
    start_node->h = heuristic(start_node, goal_node);
    start_node->f = start_node->g + start_node->h;
    start_node->set_flag = IN_OPENSET;
    start_node->it_open = openSet.insert(std::make_pair(start_node->f, start_node));

    vector<Node3D*> neighbors;
    vector<double> costs;
    vector<Vector3d> neighborState;
    vector<Vector2d> neighborVW;
    while (!openSet.empty())
    {
        result.iterations++;
        auto current = openSet.begin()->second;
        if (isFinish(current, goal_node))
        {
            result.success = true;
            result.cost = current->g;
            //如果不需要路径，直接返回
            if(!path_flag)
            {
                break;
            }
            while (current != nullptr)
            {
                result.path.push_back(current->state);
                result.controls.push_back(current->u);
                current = current->father;
            }
            std::reverse(result.path.begin(), result.path.end());
            std::reverse(result.controls.begin(), result.controls.end());
            //node中的u记录的是上一个节点到当前节点的控制，因此需要删除第一个控制，并在最后补0
            result.controls.erase(result.controls.begin());
            result.controls.push_back(Vector2d(0, 0));
            break; 
        }
        openSet.erase(current->it_open);
        current->set_flag = IN_CLOSESET;
        closeSet.push_back(current);

        
        getNeighbors(current, neighbors, costs, neighborState, neighborVW);
        for (int i = 0; i < neighbors.size(); i++)
        {
            auto neighbor = neighbors[i];
            double tentative_g = current->g + costs[i];
            if (neighbor->set_flag != IN_OPENSET)
            {
                neighbor->h = heuristic(neighbor, goal_node);
                neighbor->g = tentative_g;
                neighbor->f = neighbor->g + neighbor->h;
                neighbor->father = current;
                neighbor->set_flag = IN_OPENSET;
                neighbor->it_open = openSet.insert(std::make_pair(neighbor->f, neighbor));
                neighbor->state = neighborState[i];
                neighbor->u = neighborVW[i];
            }
            else if (tentative_g < neighbor->g)
            {
                neighbor->g = tentative_g;
                neighbor->h = heuristic(neighbor, goal_node);
                neighbor->f = neighbor->g + neighbor->h;
                neighbor->father = current;
                openSet.erase(neighbor->it_open);
                neighbor->it_open = openSet.insert(std::make_pair(neighbor->f, neighbor));
                neighbor->state = neighborState[i];
                neighbor->u = neighborVW[i];
            }
        }
    }
    if (result.success && path_flag)
    {
        _generateTrace();
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    result.planTime = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return result;
}

//获取邻居节点和对应的状态
//node:当前节点
//neighbors:邻居节点
//neighborState:邻居节点状态, [弧长(代价)，割线长度，割线与切线夹角，转角]
//neighborVW:邻居节点对应的速度和角速度
void HybridAStar::getNeighbors(Node3D *node, 
                               vector<Node3D*>& neighbors, 
                               vector<double>& costs,
                               vector<Vector3d>& neighborState,
                               vector<Vector2d>& neighborVW)
{
    neighbors.clear();
    costs.clear();
    neighborState.clear();
    neighborVW.clear();
    for (int i = 0; i < neighborList.size(); i++)
    {
        auto & d = neighborList[i];
        //x,y,theta的变化
        Vector3d dstate;
        dstate(0) = d(1) * std::cos(node->state(2) + d(2));
        dstate(1) = d(1) * std::sin(node->state(2) + d(2));
        dstate(2) = d(3);
        //新状态
        Vector3d newState = node->state + dstate;
        if (newState(2) > 2*M_PI )
        {
            newState(2) = newState(2) - 2 * (M_PI);
        }
        else if (newState(2) < 0)
        {
            newState(2) = newState(2) + 2 * (M_PI);
        }
        //保护
        // newState = newState.cwiseMax(Vector3d(0, 0, 0)).cwiseMin(map_max);
        //新状态对应的节点
        int index = getNodeIndex(newState);
        if (index < 0)
        {
            continue;
        }
        auto neighbor = nodes[index];
        if (neighbor != nullptr && !isCollision(newState) && neighbor->set_flag != IN_CLOSESET)
        {
            neighborState.push_back(newState);
            neighborVW.push_back(vwList[i]);
            neighbors.push_back(neighbor);
            costs.push_back(d(0));
        }
    }
}

void HybridAStar::_generateNeighborList(void)
{
    double v_ = max_v / step_v;
    double w_ = max_w / step_w;
    for (int i = -step_v; i <= step_v; i++)
    {
        for (int j = -step_w; j <= step_w; j++)
        {
            if (i == 0 && j == 0)
            {
                continue;
            }
            double v = i * v_;
            double w = j * w_;
            
            Vector4d neighbor;
            //转角
            neighbor(3) = w * dt;
            //割线与切线夹角
            neighbor(2) = neighbor(3) / 2;
            //割线长度
            if (j == 0)
            {
                neighbor(1) = v * dt;
            }
            else
            {
                double R = v / w;
                double L = std::sqrt(2*R*R*(1 - std::cos(w * dt)));
                neighbor(1) = L;
            }
            
            // neighbor(0) = dt;
            // 弧长，作为代价
            neighbor(0) = abs(v*dt);
            // 原地转向，按移动能力换算代价
            if (v == 0)
            {
                neighbor(0) = abs(neighbor(3))*max_v/max_w;
            }
            if (v < 0)
            {
                neighbor(0) *= 2;
            }
            vwList.push_back(Vector2d(v, w));
            neighborList.push_back(neighbor);

            
        }
    }
}

void HybridAStar::_generateTrace(void)
{
    for (int i = 0; i < result.path.size(); i++)
    {
        double v = result.controls[i](0);
        double w = result.controls[i](1);
        Vector3d state = result.path[i];
        for (int j = 0; j < trace_step; j++)
        {
            double ddt = trace_dt * j;
            double theta = w * ddt;//转角
            double theta_2 = theta / 2;//割线与切线夹角
            double L = 0;//弦长
            if (w == 0)
            {
                L = v * ddt;
            }
            else
            {
                double R = v / w;
                L = std::sqrt(2*R*R*(1 - std::cos(theta)));
            }
            Vector3d dstate(L * cos(state(2) + theta_2), L * sin(state(2) + theta_2), theta);
            Vector3d newState = state + dstate;
            result.trace.push_back(newState);
            result.trace_controls.push_back(result.controls[i]);
        }
    }
}

void HybridAStar::reset(void)
{
    for (auto p : openSet)
    {
        (p.second)->set_flag = NOT_VISITED;
        (p.second)->father = nullptr;
    }
    for (auto p : closeSet)
    {
        p->set_flag = NOT_VISITED;
        p->father = nullptr;
    }
    openSet.clear();
    closeSet.clear();
    result = PlanResult();
}
