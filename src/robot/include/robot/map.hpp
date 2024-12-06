#pragma once

#include <iostream>
#include <Eigen/Eigen>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
// #include "matplotlibcpp.h"
// namespace plt = matplotlibcpp;
using namespace std;
using namespace Eigen;

// 用于多机器人任务分配的2D地图类
class Map_2D
{
public:
    double resolution_x; // 地图分辨率
    double resolution_y;
    int n_x; // 地图尺寸
    int n_y;
    int n_starts; // 起点数量
    int n_tasks; // 任务数量
    int n_obstacles; // 障碍物数量
    int n_ob_points; // 障碍物顶点数

    //实际坐标
    vector<Vector3d> starts; // 起点, x, y, theta
    vector<Vector3d> tasks; // 任务, x, y, theta 但是theta不用
    vector<uint8_t> task_status; // 任务状态, 1为已完成
    vector<vector<Vector2d>> obstacles; // 障碍物

    //栅格地图
    uint8_t* grid_map; // 0:可通行, 1:障碍物, [x * n_y + y]

    ~Map_2D()
    {
        delete[] grid_map;
    }
    void init(double resolution_x, double resolution_y, int n_x, int n_y, int n_starts, int n_tasks, int n_obstacles, int n_ob_points);

    // 手动输入地图
    void input_map(const vector<Vector3d> &starts_, const vector<Vector3d> &tasks_, const vector<vector<Vector2d>> &obstacles_);

    // 显示地图
    // void show_map();
    // void show_grid_map();
};