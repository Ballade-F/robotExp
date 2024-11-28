#include "map.hpp"


void Map_2D::input_map(const vector<Vector3d> &starts_, const vector<Vector3d> &tasks_, const vector<vector<Vector2d>> &obstacles_)
{
    starts = starts_;
    tasks = tasks_;
    obstacles = obstacles_;
    //由多边形障碍物生成栅格地图
    for (int i = 0; i < n_obstacles; i++)
    {
        vector<Vector2i> obstacle;
        int min_y = n_y;
        int max_y = 0;
        for (int j = 0; j < obstacles[i].size(); j++)
        {
            Vector2d point = obstacles[i][j];
            int x_idx = min(max(static_cast<int>(point.x() / resolution_x), 0), n_x - 1);
            int y_idx = min(max(static_cast<int>(point.y() / resolution_y), 0), n_y - 1);
            obstacle.push_back(Vector2i(x_idx, y_idx));
            min_y = min(min_y, y_idx);
            max_y = max(max_y, y_idx);
        }
        if (min_y == max_y)
        {
            continue;
        }
        vector<Vector4d> edge_table;
        for (int j = 0; j < obstacle.size(); j++)
        {
            Vector2i point1 = obstacle[j];
            Vector2i point2 = obstacle[(j + 1) % n_ob_points];
            if (point1.y() == point2.y())
            {
                continue;
            }
            //保证point1.y() < point2.y()
            if (point1.y() > point2.y())
            {
                swap(point1, point2);
            }
            edge_table.push_back(Vector4d(point1.y(), point2.y(), point1.x(), (point2.x() - point1.x()) / (point2.y() - point1.y())));
        }
        for (int y = min_y+1; y < max_y+1; y++)
        {
            vector<Vector2i> intersections;
            for (int j = 0; j < edge_table.size(); j++)
            {
                Vector4d edge = edge_table[j];
                if (y >= edge.x() && y < edge.y())
                {
                    intersections.push_back(Vector2i(edge.z() + edge.w() * (y - edge.x()), y));
                }
            }
            sort(intersections.begin(), intersections.end(), [](Vector2i a, Vector2i b) { return a.x() < b.x(); });
            for (int j = 0; j < intersections.size(); j += 2)
            {
                int x1 = intersections[j].x();
                int x2 = intersections[j + 1].x();
                for (int x = x1; x < x2; x++)
                {
                    grid_map[x*n_y + y] = 1;
                }
            }
        }

        
        
    }
}

void Map_2D::show_map()
{
    plt::cla();
    plt::xlim(0.0, static_cast<double>(n_x) * resolution_x);
    plt::ylim(0.0, static_cast<double>(n_y) * resolution_y);
    for (int i = 0; i < n_starts; i++)
    {
        plt::plot({starts[i].x()} , {starts[i].y()} , "ro");
    }
    for (int i = 0; i < n_tasks; i++)
    {
        plt::plot({tasks[i].x()} , {tasks[i].y()} , "go");
    }
    for (int i = 0; i < n_obstacles; i++)
    {
        vector<Vector2d> obstacle = obstacles[i];
        for (int j = 0; j < n_ob_points; j++)
        {
            Vector2d point1 = obstacle[j];
            Vector2d point2 = obstacle[(j + 1) % n_ob_points];
            plt::plot({point1.x(), point2.x()}, {point1.y(), point2.y()}, "b-");
        }
    }
    // plt::show();
}

void Map_2D::show_grid_map()
{
    plt::cla();
    uint8_t grid_reverse[n_x * n_y];
    for (int i = 0; i < n_x; i++)
    {
        for (int j = 0; j < n_y; j++)
        {
            grid_reverse[j * n_x + i] = 255-grid_map[j * n_x + i];
        }
    }
    plt::imshow(grid_reverse, n_x, n_y, 1, {{"cmap", "gray"}});
    plt::show();
}
