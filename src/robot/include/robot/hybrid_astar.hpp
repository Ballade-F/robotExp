#pragma once

#include <iostream>
#include <Eigen/Eigen>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <chrono>

using namespace std;
using namespace Eigen;

enum NodeSetFlag
{
    NOT_VISITED,
    IN_OPENSET,
    IN_CLOSESET
};

class Node3D
{
public:
    Vector3d state;
    Vector3i index;
    Vector2d u;//速度和角速度
    double g;//目前代价值
    double h;//启发式值
    double f;//f=g+h
    NodeSetFlag set_flag = NOT_VISITED;//0:未加入open set和close set 1:加入open set 2:加入close set

    Node3D* father = nullptr;//祖先节点

    std::multimap<double, Node3D*>::iterator it_open;//节点在open set中的迭代器

    
    bool operator==(const Node3D& node) const {
        return index == node.index;
    }
};

class PlanResult
{
public:
    bool success = false;
    int iterations = 0;
    double planTime = 0;
    double cost = 0;
    std::vector<Vector3d> path;
    std::vector<Vector2d> controls;
    vector<Vector3d> trace;
    vector<Vector2d> trace_controls;
};

class HybridAStar
{
public:
    //参数
    double finish_radius = 0.1;
    double max_v = 1.0;
    double max_w = 1.0;
    int step_v = 1;
    int step_w = 1;
    double dt = 0.5;
    double trace_dt = 0.1;

    int trace_step = 5; //相邻path点之间的轨迹点数

    Vector3d resolution;//x,y,theta的分辨率
    Vector3i grid_size;//x,y,theta的数量
    Vector3d map_max;//地图最大值
    vector<Node3D*> nodes;//所有节点
    vector<Node3D*> closeSet;//close set
    uint8_t* grid_map;
    std::multimap<double, Node3D*> openSet;
    PlanResult result;

    //开关标志
    bool path_flag = true;//是否需要路径

private:
    //提前算好离散vw对应的弧长（代价），割线长度，割线与切线夹角，转角
    vector<Vector2d> vwList;
    vector<Vector4d> neighborList;

    // vector<Vector2d> trace_vwList;
    // vector<Vector4d> trace_stateList;

    void _generateNeighborList(void);
    void _generateTrace(void);


public:
    // HybridAStar(Vector3d resolution, Vector3i grid_size, uint8_t* grid_map, double max_v = 1.0, double max_w = 1.0, int step_v = 1, int step_w = 1, 
    //             int trace_step_ = 5, double dt = 0.1, double finish_radius = 0.1, bool path_flag_ = true)
    //         : resolution(resolution), grid_size(grid_size), grid_map(grid_map), finish_radius(finish_radius), 
    //           max_v(max_v), max_w(max_w), step_v(step_v), step_w(step_w), dt(dt), path_flag(path_flag_), trace_step(trace_step_)
    // {
    //     map_max = Vector3d(grid_size[0]*resolution[0],grid_size[1]*resolution[1],2*M_PI);
    //     nodes.resize(grid_size[0]*grid_size[1]*grid_size[2],nullptr);
    //     for (int i = 0; i < grid_size[0]; i++)
    //     {
    //         for (int j = 0; j < grid_size[1]; j++)
    //         {
    //             for (int k = 0; k < grid_size[2]; k++)
    //             {
    //                 Node3D* node = new Node3D;
    //                 node->index = Vector3i(i,j,k);
    //                 node->state = Vector3d(i*resolution[0],j*resolution[1],k*resolution[2]);
    //                 nodes[i*grid_size[1]*grid_size[2]+j*grid_size[2]+k] = node;
    //             }
    //         }
    //     }
    //     _generateNeighborList();
    // }

    ~HybridAStar()
    {
        for(auto node:nodes)
        {
            delete node;
        }
    }
    void init(Vector3d resolution, Vector3i grid_size, uint8_t* grid_map, double max_v, double max_w, int step_v, int step_w, 
                int trace_step_, double dt, double finish_radius, bool path_flag_ = true);

    PlanResult& plan(Vector3d start, Vector3d goal);

    void reset(void);

    void getNeighbors(Node3D *node, 
                  vector<Node3D*>& neighbors, 
                  vector<double>& costs,
                  vector<Vector3d>& neighborState,
                  vector<Vector2d>& neighborVW);
    

    inline int getNodeIndex(Vector3d state)
    {
        //做保护
        // if (state[0] < 0 || state[0] >= map_max[0] || state[1] < 0 || state[1] >= map_max[1])
        // {
        //     return -1;
        // }
        state = state.cwiseMax(Vector3d(0,0,0)).cwiseMin(map_max);
        Vector3i index = Vector3i(state[0]/resolution[0],state[1]/resolution[1],state[2]/resolution[2]);
        index = index.cwiseMin(grid_size - Vector3i(1, 1, 1)).cwiseMax(Vector3i(0, 0, 0));
        return index[0]*grid_size[1]*grid_size[2]+index[1]*grid_size[2]+index[2];
    }
    
    inline int getGridMapIndex(Vector3d state)
    {
        //做保护
        state = state.cwiseMax(Vector3d(0,0,0)).cwiseMin(map_max);
        // if (state[0] < 0 || state[0] >= map_max[0] || state[1] < 0 || state[1] >= map_max[1])
        // {
        //     return -1;
        // }
        Vector3i index = Vector3i(state[0]/resolution[0],state[1]/resolution[1],state[2]/resolution[2]);
        index = index.cwiseMin(grid_size - Vector3i(1, 1, 1)).cwiseMax(Vector3i(0, 0, 0));
        return index[0]*grid_size[1] + index[1];
    }

    inline bool isCollision(Vector3d state)
    {
        int index = getGridMapIndex(state);
        return grid_map[index] == 1;
    }

    inline double heuristic(Node3D* node1, Node3D* node2)
    {
        return (node1->state.block<2,1>(0,0)-node2->state.block<2,1>(0,0)).norm();
        // double dist = (node1->state.block<2,1>(0,0)-node2->state.block<2,1>(0,0)).norm();
        // double angle = std::abs(node1->state[2]-node2->state[2]);
        // double t1 = dist / max_v;
        // double t2 = angle / max_w;
        // return std::max(t1,t2);
    }

    inline bool isFinish(Node3D* node, Node3D* goal_node)
    {
        return (node->state.block<2,1>(0,0)-goal_node->state.block<2,1>(0,0)).norm()<finish_radius;
    }
};