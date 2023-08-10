#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.
  
#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors: 
# marcos-pereira (https://github.com/marcos-pereira)


#!/usr/bin/env python
from rtree import index
from RRT import RRT
from RRTStar import RRTStar
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
    
    rrt_planner = RRT(x_init,
                    x_goal,
                    goal_radius,
                    steer_delta,
                    scene_map,
                    num_nodes)
    
    rrt_planner.run()

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
                    num_nodes)
    
    rrt_planner.run()


if __name__ == '__main__':

    main()