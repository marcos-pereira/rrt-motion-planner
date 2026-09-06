#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.
  
#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors: 
# marcos-pereira (https://github.com/marcos-pereira)

#!/usr/bin/env python
from rtree import index
# from RRTPlannerMapPyglet import RRTPlanner
# from RRTStarPlannerMapPyglet import RRTStarPlanner
from RRT import RRT
from RRTStar import RRTStar
from SimpleDeltaSteering import SimpleDeltaSteering
from Map import load_map
from PlanDrawer import PlanDrawer
import sys

def main():
    
    # Check if the correct number of command-line arguments are provided
    if len(sys.argv) < 4:
        print("Usage: python3 main.py <map_name.png> <steer_step_size> <goal_radius> <max_num_nodes_in_tree> <x_init> <y_init> <x_goal> <y_goal> [max_planning_time_seconds] [gamma_rrt] [eta_rrt] [near_radius]")
        return

    # Access the command-line arguments starting from index 1
    arguments = sys.argv[1:]

    max_planning_time = None

    # Default RRT* tuning parameters, used unless overridden below.
    gamma_rrt = 1000
    eta_rrt = 20
    near_radius = 20

    # Print the arguments
    print("Command-line arguments:")
    for i, arg in enumerate(arguments, start=1):
        print(f"Argument {i}: {arg}")
        if i == 1:
            map_name = arg
        elif i == 2:
            steer_delta = float(arg)
        elif i == 3:
            goal_radius = int(arg)
        elif i == 4:
            num_nodes = int(arg)
        elif i == 5:
            x_init = int(arg)
        elif i == 6:
            y_init = int(arg)
        elif i == 7:
            x_goal = int(arg)
        elif i ==8:
            y_goal = int(arg)
        elif i == 9:
            max_planning_time = float(arg)
        elif i == 10:
            gamma_rrt = float(arg)
        elif i == 11:
            eta_rrt = float(arg)
        elif i == 12:
            near_radius = float(arg)

    init_node = (x_init, y_init)
    goal_node = (x_goal, y_goal)
    # goal_radius = 10
    # steer_delta = 15
    # num_nodes = 50000
    font_size = 25
    # map_name = 'smile1.png'

    scene_map = load_map(map_name)

    steer = SimpleDeltaSteering()

    rrt_planner = RRT(init_node,
                    goal_node,
                    goal_radius,
                    steer_delta,
                    steer,
                    scene_map,
                    num_nodes,
                    max_planning_time)
    
    map_height, map_width = scene_map.shape
    plan_drawer_rrt = PlanDrawer(map_name, map_width, map_height, font_size)
    plan_drawer_rrt.run(rrt_planner)
    
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

    rrtstar_planner = RRTStar(init_node,
                    goal_node,
                    goal_radius,
                    steer_delta,
                    steer,
                    eta_rrt,
                    gamma_rrt,
                    near_radius,
                    scene_map,
                    num_nodes,
                    max_planning_time)
    
    plan_drawer_rrtstar = PlanDrawer(map_name, map_width, map_height, font_size)
    plan_drawer_rrtstar.run_forever(rrtstar_planner)
    
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