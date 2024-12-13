from copy import deepcopy
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import json
#obstacle generator parameters
# ob_rmax_factor = 0.1
# ob_rmin_factor = 0.06
# ob_points_min = 8
# ob_points_max = 10
# n_ob_points = 16



# Store coordinates in a normalized manner
class Map():
    def __init__(self, n_obstacles:int, n_starts:int, n_tasks:int, n_x:int, n_y:int, resolution_x:float, resolution_y:float, n_ob_points:int=16):
        self.n_obstacles = n_obstacles
        self.n_ob_points = n_ob_points
        self.n_starts = n_starts
        self.n_tasks = n_tasks
        self.n_x = n_x
        self.n_y = n_y
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.x_max = n_x*resolution_x
        self.y_max = n_y*resolution_y
        self.obstacles = []
        self.starts = []
        self.tasks = []
        self.tasks_finish = []
        self.starts_grid = []
        self.tasks_grid = []
        self.grid_map = np.zeros((n_x, n_y))

    def _obstacle2grid(self):
        obstacles_back = deepcopy(self.obstacles)
        for ob_points in obstacles_back:
            ob_points[:, 0] = ob_points[:, 0] * self.n_x
            ob_points[:, 1] = ob_points[:, 1] * self.n_y

            min_x_index = int(np.min(ob_points[:, 0]))
            max_x_index = int(np.max(ob_points[:, 0]))
            min_y_index = int(np.min(ob_points[:, 1]))
            max_y_index = int(np.max(ob_points[:, 1]))

            min_y_index = max(0, min(min_y_index, self.n_y-1))
            max_y_index = max(0, min(max_y_index, self.n_y-1))

            # #debug
            # print(ob_points)
            # print(min_x_index, max_x_index, min_y_index, max_y_index)

            # Create an edge table
            edge_table = []
            for i in range(len(ob_points)):
                x1, y1 = ob_points[i]
                x2, y2 = ob_points[(i + 1) % len(ob_points)]
                if y1 != y2:
                    #ensure x1, y1 is the lower endpoint
                    if y1 > y2:
                        x1, y1, x2, y2 = x2, y2, x1, y1
                    edge_table.append([y1, y2, x1, (x2 - x1) / (y2 - y1)])

            
            if min_y_index == max_y_index:
                # self.grid_map[min_x_index:max_x_index + 1, min_y_index] = 1
                continue

            #if the x_axis ray crosses the polygon, the grids both up and down the ray should be filled
            for y in range(min_y_index + 1, max_y_index + 1):
                active_edge_table = []
                for edge in edge_table:
                    if edge[0] <= y and edge[1] > y:
                        active_edge_table.append(edge)
                # Sort active edge table by x-coordinate
                active_edge_table.sort(key=lambda edge: edge[2]+edge[3]*(y-edge[0]))
                # fill pixels between pairs of intersections
                for i in range(0, len(active_edge_table), 2):
                    x_start = int(active_edge_table[i][2]+active_edge_table[i][3]*(y-active_edge_table[i][0]))
                    x_end = int(active_edge_table[i + 1][2]+active_edge_table[i+1][3]*(y-active_edge_table[i+1][0]))
                    x_start = max(0,min(x_start, self.n_x-1))
                    x_end = max(0,min(x_end, self.n_x-1))
                    self.grid_map[x_start:x_end+1, y-1:y+1] = 1

    # input: true coordinates
    def isObstacle(self, x:float, y:float):
        x = np.maximum(x, 0)
        x = np.minimum(x, self.x_max)
        y = np.maximum(y, 0)
        y = np.minimum(y, self.y_max)
        x_index = np.floor(x/self.resolution_x).astype(int)
        y_index = np.floor(y/self.resolution_y).astype(int)
        return self.grid_map[x_index, y_index] == 1
    
    # input: normalized coordinates
    def _isObstacle(self, x, y):
        x = np.maximum(x, 0)
        x = np.minimum(x, 1)
        y = np.maximum(y, 0)
        y = np.minimum(y, 1)
        x_index = np.floor(x*self.n_x).astype(int)
        y_index = np.floor(y*self.n_y).astype(int)
        return self.grid_map[x_index, y_index] == 1
    
    def true2grid(self, point)->tuple:
        x_index = int(point[0]/self.resolution_x)
        y_index = int(point[1]/self.resolution_y)
        x_index = max(0, min(x_index, self.n_x-1))
        y_index = max(0, min(y_index, self.n_y-1))
        return (x_index, y_index)
    
    # input: true coordinates
    def setStartTask(self, start:np.ndarray, tasks:np.ndarray):
        start_ = start.copy()
        start_[:, 0] = start[:, 0] / (self.n_x*self.resolution_x)
        start_[:, 1] = start[:, 1] / (self.n_y*self.resolution_y)
        start_ = np.maximum(start_, 0)
        start_ = np.minimum(start_, 1)
        self.starts = start_

        tasks_ = tasks.copy()
        tasks_[:, 0] = tasks[:, 0] / (self.n_x*self.resolution_x)
        tasks_[:, 1] = tasks[:, 1] / (self.n_y*self.resolution_y)
        tasks_ = np.maximum(tasks_, 0)
        tasks_ = np.minimum(tasks_, 1)
        self.tasks = tasks_

        self.tasks_finish = [False for _ in range(self.n_tasks)]

        self.starts_grid = np.zeros((self.n_starts, 2), dtype=int)
        self.tasks_grid = np.zeros((self.n_tasks, 2), dtype=int)
        self.starts_grid[:,0] = np.floor(start_[:,0]*self.n_x).astype(int)
        self.starts_grid[:,0] = max(0, min(self.starts_grid[:,0], self.n_x-1))
        self.starts_grid[:,1] = np.floor(start_[:,1]*self.n_y).astype(int)
        self.starts_grid[:,1] = max(0, min(self.starts_grid[:,1], self.n_y-1))
        self.tasks_grid[:,0] = np.floor(tasks_[:,0]*self.n_x).astype(int)
        self.tasks_grid[:,0] = max(0, min(self.tasks_grid[:,0], self.n_x-1))
        self.tasks_grid[:,1] = np.floor(tasks_[:,1]*self.n_y).astype(int)
        self.tasks_grid[:,1] = max(0, min(self.tasks_grid[:,1], self.n_y-1))

    # input: true coordinates
    def setObstacles(self, obstacles: list, start: np.ndarray, tasks: np.ndarray):
        for ob_points in obstacles:
            ob_points[:, 0] = ob_points[:, 0] / (self.n_x*self.resolution_x)
            ob_points[:, 1] = ob_points[:, 1] / (self.n_y*self.resolution_y)
            self.obstacles.append(ob_points)

        self.setStartTask(start, tasks)

        self._obstacle2grid()


          
        
    # input: true coordinates(default) or norm coordinates, & id of task
    # if point and task are in the same grid, return True
    def checkTaskFinish(self, x:float, y:float, task_id:int,norm_flag=False):
        x_index = -1
        y_index = -1
        if norm_flag:
            x = np.maximum(x, 0)
            x = np.minimum(x, 1)
            y = np.maximum(y, 0)
            y = np.minimum(y, 1)
            x_index = np.floor(x*self.n_x).astype(int)
            y_index = np.floor(y*self.n_y).astype(int)
        else:
            x = np.maximum(x, 0)
            x = np.minimum(x, self.x_max)
            y = np.maximum(y, 0)
            y = np.minimum(y, self.y_max)
            x_index = np.floor(x/self.resolution_x).astype(int)
            y_index = np.floor(y/self.resolution_y).astype(int)
        if self.tasks_grid[task_id][0] == x_index and self.tasks_grid[task_id][1] == y_index:
            self.tasks_finish[task_id] = True
            return True
        return False

    # return normalized coordinates of obstacles, starts and tasks in -1 to 1
    def dataForDL(self):
        obstacles = []
        for ob_points in self.obstacles:
            ob_points = (ob_points - 0.5)*2
            obstacles.append(ob_points)
        starts = (self.starts - 0.5)*2
        tasks = (self.tasks - 0.5)*2
        return obstacles, starts, tasks

    def plot(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for ob_points in self.obstacles:
            ax.fill(ob_points[:, 0], ob_points[:, 1], 'r')
        for i in range(self.n_starts):
            ax.scatter(self.starts[i, 0], self.starts[i, 1], c='b')
            plt.text(self.starts[i, 0], self.starts[i, 1], str(i))
        for i in range(self.n_tasks):
            ax.scatter(self.tasks[i, 0], self.tasks[i, 1], c='r')
            plt.text(self.tasks[i, 0], self.tasks[i, 1], str(i))
        # ax.scatter(self.starts[:, 0], self.starts[:, 1], c='b',text=i) 
        # ax.scatter(self.tasks[:, 0], self.tasks[:, 1], c='r',text=i) 
        plt.show()

    def plotGrid(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.n_x)
        ax.set_ylim(0, self.n_y)
        img = 255-self.grid_map*255
        img = img.transpose()
        ax.imshow(img, cmap='gray')
        # for ob_points in self.obstacles:
        #     ax.fill(ob_points[:, 0]*self.n_x, ob_points[:, 1]*self.n_y, 'r')
        plt.show()

    def setObstacleExp(self, rng:np.random.Generator):
        # rng = np.random.default_rng(seed)
        ob_len = 0.071
        center_points = rng.uniform(0, 1, (self.n_obstacles, 2))
        ob_theta = rng.uniform(0, 0.5*np.pi, (self.n_obstacles))
        for i in range(self.n_obstacles):
            ob_points = np.zeros((4, 2)) #正方形
            for j in range(4):
                ob_points[j, 0] = center_points[i, 0] + ob_len*np.cos(ob_theta[i]+j*np.pi/2)
                ob_points[j, 1] = center_points[i, 1] + ob_len*np.sin(ob_theta[i]+j*np.pi/2)
            ob_points = np.maximum(ob_points, 0)
            ob_points = np.minimum(ob_points, 1)
            self.obstacles.append(ob_points)


        self._obstacle2grid()

        while(True):
            self.starts = rng.uniform(0, 1, (self.n_starts, 2))
            self.tasks = rng.uniform(0, 1, (self.n_tasks, 2))
            if True not in self._isObstacle(self.starts[:, 0], self.starts[:, 1])  and True not in self._isObstacle(self.tasks[:, 0], self.tasks[:, 1]) :
                break

        # self.tasks_finish = [False for _ in range(self.n_tasks)]

        # self.starts_grid = np.zeros((self.n_starts, 2), dtype=int)
        # self.tasks_grid = np.zeros((self.n_tasks, 2), dtype=int)
        # self.starts_grid[:,0] = np.floor(self.starts[:,0]*self.n_x).astype(int)
        # self.starts_grid[:,1] = np.floor(self.starts[:,1]*self.n_y).astype(int)
        # self.tasks_grid[:,0] = np.floor(self.tasks[:,0]*self.n_x).astype(int)
        # self.tasks_grid[:,1] = np.floor(self.tasks[:,1]*self.n_y).astype(int)
        
  

    #输入正方形障碍的中心点，角度，边长，起点，任务点，都是真实值
    def setExp(self, center_points:np.ndarray, ob_theta:np.ndarray, ob_len:float, start:np.ndarray, tasks:np.ndarray):
        len_slant = ob_len/np.sqrt(2)
        for i in range(self.n_obstacles):
            ob_points = np.zeros((4, 2)) #正方形的真实坐标
            for j in range(4):
                ob_points[j, 0] = center_points[i, 0] + len_slant*np.cos(ob_theta[i]+j*np.pi/2 + np.pi/4)
                ob_points[j, 1] = center_points[i, 1] + len_slant*np.sin(ob_theta[i]+j*np.pi/2 + np.pi/4)
            ob_points[:, 0] = ob_points[:, 0] / (self.n_x*self.resolution_x)
            ob_points[:, 1] = ob_points[:, 1] / (self.n_y*self.resolution_y)
            ob_points = np.maximum(ob_points, 0)
            ob_points = np.minimum(ob_points, 1)
            self.obstacles.append(ob_points)
        self._obstacle2grid()
        
        start_ = start.copy()
        start_[:, 0] = start[:, 0] / (self.n_x*self.resolution_x)
        start_[:, 1] = start[:, 1] / (self.n_y*self.resolution_y)
        start_ = np.maximum(start_, 0)
        start_ = np.minimum(start_, 1)
        self.starts = start_

        tasks_ = tasks.copy()
        tasks_[:, 0] = tasks[:, 0] / (self.n_x*self.resolution_x)
        tasks_[:, 1] = tasks[:, 1] / (self.n_y*self.resolution_y)
        tasks_ = np.maximum(tasks_, 0)
        tasks_ = np.minimum(tasks_, 1)
        self.tasks = tasks_
        
    def saveExpMap(self, file_dir:str):
        os.makedirs(file_dir, exist_ok=True)
        # Save map information to a JSON file
        batch_info = {
            "batch_size": 1,
            "n_robot": self.n_starts,
            "n_task": self.n_tasks,
            "n_obstacle": self.n_obstacles,
            "ob_points": 4,
            "n_x": self.n_x,
            "n_y": self.n_y,
            "resolution_x": self.resolution_x,
            "resolution_y": self.resolution_y
        }
        with open(os.path.join(file_dir, "batch_info.json"), "w") as json_file:
            json.dump(batch_info, json_file, indent=4)
            
        # Save map information to a CSV file
        with open(os.path.join(file_dir, f"info.csv"), "w", newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Type", "x", "y"])
            for start in map.starts:
                writer.writerow([-1, start[0], start[1]])
            for task in map.tasks:
                writer.writerow([0, task[0], task[1]])
            for ob_idx, ob in enumerate(map.obstacles):
                for point in ob:
                    writer.writerow([ob_idx+1, point[0], point[1]])

if __name__ == '__main__':
    map = Map(10, 2, 5, 116, 116, 0.05, 0.05,4)
    ob_center_grid = np.array(
        [[3,3],[3,4],[7,2],[4,5],[9,5],
         [1.5,6.5],[6.5,6.5],[4.5,8.5],[7,9],[8.5,8.5]], dtype=float)
    ob_theta = np.zeros(10)
    
    # ob_center_grid = np.array([[3.0, 3.0]], dtype=float)
    # ob_theta = np.array([0.0], dtype=float)
    
    ob_center_grid *= 0.58
    
    ob_len = 0.8
    starts = np.array([[0.5, 1.0], [3.0, 1.5]])
    tasks = np.array(
        [[6.5, 4.5], [1.5, 3.5], [8.5, 6.5], [2.0, 7.5], [5.5, 9.5]])
    tasks *= 0.58
    map.setExp(ob_center_grid, ob_theta, ob_len, starts, tasks)
    map.saveExpMap('/home/jxl3028/Desktop/wzr/robotExp/src/config/map/map_exp')
    
    # rng = np.random.default_rng(2)
    # map.setObstacleExp(rng)

    map.plot()
    map.plotGrid()



