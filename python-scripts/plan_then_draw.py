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
    
    x_init = (30, 30)
    x_goal = (30, 460)
    goal_radius = 10
    steer_delta = 15
    near_radius = 30
    num_nodes = 20000
    max_planning_time = None
    font_size = 25
    map_name = 'smile.png'

    scene_map = load_map(map_name, test=True)
    map_height, map_width = scene_map.shape

    rrt_planner = RRT(x_init,
                    x_goal,
                    goal_radius,
                    steer_delta,
                    scene_map,
                    num_nodes,
                    max_planning_time)

    path, path_cost = rrt_planner.run()
    
    plan_drawer = PlanDrawer(map_name, map_width, map_height, font_size)
    plan_drawer.draw(rrt_planner.tree_builder_, x_goal, goal_radius, path)

    gamma_rrt = 1000
    eta_rrt = 20
    near_radius = 50
    rrt_planner = RRTStar(x_init,
                    x_goal,
                    goal_radius,
                    steer_delta,
                    eta_rrt,
                    gamma_rrt,
                    near_radius,
                    scene_map,
                    num_nodes,
                    max_planning_time)

    rrt_planner.run()


if __name__ == '__main__':

    main()