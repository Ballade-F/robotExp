import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
import json
import csv
import os

import intention 
import allocation 
import greedy_allocation_lib as greedy
import ga_allocation_lib as ga

class Robot:
    def __init__(self, map_dir:str, robot_dir:str):
        
        # 读取地图
        batch_info = os.path.join(dir, "batch_info.json")
        with open(batch_info, "r") as f:
            map_info = json.load(f)

        self.n_robot = map_info["n_robot"]
        self.n_task = map_info["n_task"]
        self.n_obstacle = map_info["n_obstacle"]
        self.ob_points = map_info["ob_points"]
        self.n_x = map_info["n_x"]
        self.n_y = map_info["n_y"]
        self.resolution_x = map_info["resolution_x"]
        self.resolution_y = map_info["resolution_y"]

        self.robot_states = np.zeros((self.n_robot, 3)) # x, y, theta, xy为归一化坐标
        self.task_states = np.zeros((self.n_task, 3)) # x, y, theta 但是theta不用，xy为归一化坐标
        self.obstacle_states = np.zeros((self.n_obstacle, self.ob_points, 2)) #障碍个数，每个障碍的点数，每个点的坐标，xy为归一化坐标
        self.task_finished = np.zeros((self.n_task), dtype=int)

        map_data_dir = os.path.join(map_dir, "info.csv")
        with open(map_data_dir, "r") as f:
            reader = csv.reader(f)
            for idx, row in enumerate(reader):
                #表头
                if idx == 0:
                    continue
                if idx <= self.n_robot:
                    self.robot_states[idx-1,:2] = row[1:] # x, y
                elif idx <= self.n_robot+self.n_task:
                    self.task_states[idx-self.n_robot-1,:2] = row[1:] # x, y
                else:
                    idx_ob = int(row[0])-1
                    idx_point = (idx-self.n_robot-self.n_task-1)-idx_ob*self.ob_points
                    self.obstacle_states[idx_ob, idx_point, :] = row[1:]

        # 读取机器人信息
        with open(robot_dir, "r") as f:
            robot_info = json.load(f)
        self.robot_id = robot_info["robot_id"]
        self.device = robot_info["device"]
        self.buffer_size = robot_info["buffer_size"]
        intention_model_dir = robot_info["intention_model_dir"]
        allocation_model_dir = robot_info["allocation_model_dir"]

        #TODO: 如果intention效果不好，考虑使用mask盖住已经完成的任务
        self.intention_judgment = intention.IntentionNet(ob_points=self.ob_points,r_points = self.buffer_size,device=self.device)
        self.intention_judgment.load_state_dict(torch.load(intention_model_dir, map_location=self.device))
        self.intention_judgment.config({'n_robot':self.n_robot, 
                                        'n_task':self.n_task, 
                                        'n_obstacle':self.n_obstacle})
        
        self.allocation_network = allocation.AllocationNet(ob_points=self.ob_points,device=self.device)
        



    # 用感知信息更新状态
    def updateState(self, robot_states, task_states,task_finished):
        '''
        robot_states: ndarray((n_robot,3)), 真实坐标  
        task_states: ndarray((n_task,3))，真实坐标 
        task_finished: ndarray((n_task),int)，是否完成   
        '''

    # 决策
    def decision(self, algorithm:int, pre_allocation:np.ndarray):
        '''
        algorithm: int, 0表示贪心算法，1表示遗传算法,2表示网络模型
        pre_allocation: ndarray((n_robot)), 先验分配
        '''
        