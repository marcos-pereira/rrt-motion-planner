#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.
  
#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors: 
# marcos-pereira (https://github.com/marcos-pereira)


#!/usr/bin/env python
import sys

from rtree import index
from RRT import RRT
from RRTStar import RRTStar
from Map import load_map
from PlanDrawer import PlanDrawer

def main():

    # Default parameters, used for any command-line argument left out.
    map_name = 'smile.png'
    steer_delta = 15
    goal_radius = 10
    num_nodes = 20000
    x_init_x, x_init_y = 30, 30
    x_goal_x, x_goal_y = 30, 460
    max_planning_time = 15

    # Default RRT* tuning parameters, used unless overridden below.
    gamma_rrt = 1000
    eta_rrt = 20
    near_radius = 50

    arguments = sys.argv[1:]

    if arguments and arguments[0] in ('-h', '--help'):
        print("Usage: python3 plan_then_draw.py [map_name.png] [steer_step_size] [goal_radius] "
              "[max_num_nodes_in_tree] [x_init] [y_init] [x_goal] [y_goal] [max_planning_time_seconds] "
              "[gamma_rrt] [eta_rrt] [near_radius]")
        print("Any argument left out keeps its default value.")
        return

    if arguments:
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
            x_init_x = int(arg)
        elif i == 6:
            x_init_y = int(arg)
        elif i == 7:
            x_goal_x = int(arg)
        elif i == 8:
            x_goal_y = int(arg)
        elif i == 9:
            max_planning_time = float(arg)
        elif i == 10:
            gamma_rrt = float(arg)
        elif i == 11:
            eta_rrt = float(arg)
        elif i == 12:
            near_radius = float(arg)

    x_init = (x_init_x, x_init_y)
    x_goal = (x_goal_x, x_goal_y)
    font_size = 25

    scene_map = load_map(map_name, test=True)
    map_height, map_width = scene_map.shape

    rrt_planner = RRT(x_init,
                    x_goal,
                    goal_radius,
                    steer_delta,
                    scene_map,
                    num_nodes,
                    max_planning_time)

    path, path_cost = rrt_planner.plan()

    plan_drawer_rrt = PlanDrawer(map_name, map_width, map_height, font_size)
    plan_drawer_rrt.draw(rrt_planner.tree_builder_, x_goal, goal_radius, path)

    rrtstar_planner = RRTStar(x_init,
                    x_goal,
                    goal_radius,
                    steer_delta,
                    eta_rrt,
                    gamma_rrt,
                    near_radius,
                    scene_map,
                    num_nodes,
                    max_planning_time)

    path, path_cost = rrtstar_planner.plan()

    plan_drawer_rrtstar = PlanDrawer(map_name, map_width, map_height, font_size)
    plan_drawer_rrtstar.draw_final(rrtstar_planner, path, path_cost)

    # Keep both windows open (press escape in either to close it) instead of exiting
    # immediately and closing them as soon as the trees are drawn.
    while plan_drawer_rrt.stop_drawing_ == 0 or plan_drawer_rrtstar.stop_drawing_ == 0:
        plan_drawer_rrt.dispatch_events()
        plan_drawer_rrtstar.dispatch_events()


if __name__ == '__main__':

    main()