#pragma once

#include "torch/script.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <memory>
#include <chrono>
#include <string>
#include <Eigen/Eigen>
#include "ring_buffer.hpp"

using namespace std;
using namespace Eigen;

class IntentionResult
{
public:
	vector<int> intention_id;
	vector<double> intention_prob;
	double intention_time;//ms
};

class AllocationResult
{
public:
	vector<int> allocation;
	double allocation_time;//ms
};

class Network
{
public:
    //TODO: 后续设为参数
    string device_string;
    int batch_size=1;
    int n_robot;
    int n_task;
    int n_obstacle;
    int ob_point;
    int r_point;
    double x_max;
    double y_max;
    string allocation_model_path;
    string intention_model_path;

    //libtorch
    torch::Device device = torch::Device(torch::kCPU);
    torch::jit::script::Module allocation_model;
    torch::jit::script::Module intention_model;
    vector<torch::jit::IValue> allocation_inputs;
    vector<torch::jit::IValue> intention_inputs;
    torch::Tensor obstacles;
    // vector<vector<Vector2d>> obstacles;

    //调试
    // double intention_time = 0; //ms
    // double allocation_time = 0; //ms

    IntentionResult intention_result;
    AllocationResult allocation_result;


    Network(double x_max_, double y_max_, int n_robot_, int n_task_, int n_obstacle_, int ob_point_, int r_point_,
            string allocation_model_path_, string intention_model_path_, string device_string_, string map_csv_path_);

    const AllocationResult& getAllocation(const vector<Vector3d>& robot_states_, const vector<Vector3d>& task_states_, 
                               vector<uint8_t> task_finished_,  vector<int> pre_allocation_);
    const IntentionResult& getIntention(const RingBuffer<vector<Vector3d>>& robot_states_keyframe_, 
                             const vector<Vector3d>& task_states_, const vector<uint8_t>& task_finished_);
};