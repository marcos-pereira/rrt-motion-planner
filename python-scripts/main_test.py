#!/usr/bin/env python
from rtree import index
from RRTPlannerMapPyglet import RRTPlanner
from RRTStarPlannerMapPyglet import RRTStarPlanner
from Map import load_map

def main():
    x_init = (30, 30)
    x_goal = (30, 460)
    goal_radius = 10
    steer_delta = 15
    near_radius = 30
    search_window = 800 # 400, 800, 1024
    frames_steps = 1
    num_nodes = 50000
    font_size = 25
    map_name = 'smile1.png'
    
    scene_map = load_map(map_name, test=True)
    
    ## Call RRT
    rows, cols = scene_map.shape
    rrt_planner = RRTPlanner(x_init,
                             x_goal,
                             goal_radius,
                             steer_delta,
                             map_name,
                             scene_map,
                             search_window,
                             cols,
                             rows,
                             num_nodes,
                             font_size)

    rrt_planner.run()
    
    ## Call RRTStar
    rows, cols = scene_map.shape
    rrt_planner = RRTStarPlanner(x_init,
                                 x_goal,
                                 goal_radius,
                                 steer_delta,
                                 map_name,
                                 scene_map,
                                 search_window,
                                 cols,
                                 rows,
                                 num_nodes,
                                 font_size)

    rrt_planner.run()


if __name__ == '__main__':

    main()