#!/usr/bin/env python
from rtree import index
# from RRTPlannerMapPyglet import RRTPlanner
# from RRTStarPlannerMapPyglet import RRTStarPlanner
from RRT import RRT
from RRTStar import RRTStar
from Map import load_map
from PlanDrawer import PlanDrawer

def main():
    x_init = (30, 30)
    x_goal = (30, 460)
    goal_radius = 10
    steer_delta = 0.05
    near_radius = 30
    search_window = 800 # 400, 800, 1024
    frames_steps = 1
    num_nodes = 50000
    font_size = 25
    map_name = 'smile1.png'
    
    scene_map = load_map(map_name)
    
    rrt_planner = RRT(x_init,
                    x_goal,
                    goal_radius,
                    steer_delta,
                    scene_map,
                    num_nodes)
    
    map_height, map_width = scene_map.shape
    # plan_drawer = PlanDrawer(map_name, map_width, map_height, font_size)
    # plan_drawer.run(rrt_planner)
    
    ## Call RRT
    # rows, cols = scene_map.shape
    # rrt_planner = RRTPlanner(x_init,
    #                          x_goal,
    #                          goal_radius,
    #                          steer_delta,
    #                          map_name,
    #                          scene_map,
    #                          search_window,
    #                          cols,
    #                          rows,
    #                          num_nodes,
    #                          font_size)

    # rrt_planner.run()
    
    gamma_rrt = 1000
    eta_rrt = 30
    near_radius = 50
    rrt_planner = RRTStar(x_init,
                    x_goal,
                    goal_radius,
                    steer_delta,
                    eta_rrt,
                    gamma_rrt,
                    near_radius,
                    scene_map,
                    num_nodes)
    
    plan_drawer = PlanDrawer(map_name, map_width, map_height, font_size)
    plan_drawer.run_forever(rrt_planner)
    
    ## Call RRTStar
    # rows, cols = scene_map.shape
    # rrt_planner = RRTStarPlanner(x_init,
    #                              x_goal,
    #                              goal_radius,
    #                              steer_delta,
    #                              map_name,
    #                              scene_map,
    #                              search_window,
    #                              cols,
    #                              rows,
    #                              num_nodes,
    #                              font_size)

    # rrt_planner.run()


if __name__ == '__main__':

    main()