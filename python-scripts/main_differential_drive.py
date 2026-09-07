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
from DifferentialDrivePoseSampler import DifferentialDrivePoseSampler
from DifferentialDriveSampler import DifferentialDriveSampler
from DifferentialDriveRandomControlSteering import DifferentialDriveRandomControlSteering
from DifferentialDriveCollisionChecker import DifferentialDriveCollisionChecker
from Map import load_map
from PlanDrawer import PlanDrawer
from RealVectorState import RealVectorState

def chronological_path(path, state_init):
    """ Return path (as returned by RRTPlanner.plan()/path(), ordered from the reached
    goal node back towards state_init, and excluding state_init itself) reversed into
    chronological order from state_init to the goal, with state_init prepended.
    """
    if not path:
        return []

    return [state_init.get_value()] + list(reversed(path))

def main():

    # Default parameters, used for any command-line argument left out.
    map_name = 'smile.png'
    goal_radius = 20
    num_nodes = 20000
    x_init_x, x_init_y, theta_init = 30.0, 30.0, 0.0
    x_goal_x, x_goal_y = 30.0, 460.0
    max_planning_time = 20

    # DifferentialDriveRobot parameters. wheel_radius and distance_wheels cancel out in
    # the unicycle model that DifferentialDriveRobot implements (any nonzero values give
    # the same [x_dot, y_dot, theta_dot] = [v*cos(theta), v*sin(theta), w]), so their
    # values only matter if reused elsewhere with different assumptions.
    wheel_radius = 1.0
    distance_wheels = 1.0
    sampling_time = 1.0

    # Bounds for the random [linear_velocity, angular_velocity] control sampled at
    # every RRT/RRT* iteration by DifferentialDriveSampler.
    linear_velocity_min, linear_velocity_max = 0.0, 20.0
    angular_velocity_min, angular_velocity_max = -1.0, 1.0

    # Radius of the circular footprint used by DifferentialDriveCollisionChecker and by
    # PlanDrawer.animate_differential_drive_path() to draw the robot.
    robot_radius = 8.0

    # States of the final path drawn per second by animate_differential_drive_path().
    fps = 10.0

    # Default RRT* tuning parameters, used unless overridden below.
    gamma_rrt = 1000
    eta_rrt = 20
    near_radius = 50

    arguments = sys.argv[1:]

    if arguments and arguments[0] in ('-h', '--help'):
        print("Usage: python3 main_differential_drive.py [map_name.png] [goal_radius] "
              "[max_num_nodes_in_tree] [x_init] [y_init] [x_goal] [y_goal] [max_planning_time_seconds] "
              "[robot_radius] [linear_velocity_max] [angular_velocity_max] [sampling_time] [fps]")
        print("Any argument left out keeps its default value.")
        return

    if arguments:
        print("Command-line arguments:")

    for i, arg in enumerate(arguments, start=1):
        print(f"Argument {i}: {arg}")
        if i == 1:
            map_name = arg
        elif i == 2:
            goal_radius = int(arg)
        elif i == 3:
            num_nodes = int(arg)
        elif i == 4:
            x_init_x = float(arg)
        elif i == 5:
            x_init_y = float(arg)
        elif i == 6:
            x_goal_x = float(arg)
        elif i == 7:
            x_goal_y = float(arg)
        elif i == 8:
            max_planning_time = float(arg)
        elif i == 9:
            robot_radius = float(arg)
        elif i == 10:
            linear_velocity_max = float(arg)
        elif i == 11:
            angular_velocity_max = float(arg)
        elif i == 12:
            sampling_time = float(arg)
        elif i == 13:
            fps = float(arg)

    # The goal is a position with a radius of tolerance; the goal heading is unconstrained,
    # since path_to_goal_found() only compares (x, y), so theta_goal is a placeholder.
    x_init = RealVectorState((x_init_x, x_init_y, theta_init))
    x_goal = RealVectorState((x_goal_x, x_goal_y, 0.0))
    font_size = 25

    scene_map = load_map(map_name, test=True)
    map_height, map_width = scene_map.shape

    # steer_delta is unused by DifferentialDriveRandomControlSteering (the new tree node
    # comes from a randomly sampled control, not from steering towards a target), but
    # RRT/RRTStar's constructor still requires it.
    steer_delta = sampling_time

    pose_sampler = DifferentialDrivePoseSampler(0, map_width, 0, map_height)
    control_sampler = DifferentialDriveSampler(linear_velocity_min, linear_velocity_max,
                                                angular_velocity_min, angular_velocity_max)
    steer = DifferentialDriveRandomControlSteering(control_sampler, wheel_radius, distance_wheels, sampling_time)
    collision_checker = DifferentialDriveCollisionChecker(scene_map, robot_radius)

    rrt_planner = RRT(x_init,
                    x_goal,
                    goal_radius,
                    steer_delta,
                    steer,
                    pose_sampler,
                    collision_checker,
                    num_nodes,
                    max_planning_time)

    path, path_cost = rrt_planner.plan()

    plan_drawer_rrt = PlanDrawer(map_name, map_width, map_height, font_size)
    plan_drawer_rrt.draw(rrt_planner.tree_builder_, x_goal, goal_radius, path)
    plan_drawer_rrt.animate_differential_drive_path(chronological_path(path, x_init), robot_radius, fps)

    # Wait for the user to inspect the RRT tree and press Escape before planning RRT*.
    while plan_drawer_rrt.stop_drawing_ == 0:
        plan_drawer_rrt.dispatch_events()

    rrtstar_planner = RRTStar(x_init,
                    x_goal,
                    goal_radius,
                    steer_delta,
                    steer,
                    pose_sampler,
                    eta_rrt,
                    gamma_rrt,
                    near_radius,
                    collision_checker,
                    num_nodes,
                    max_planning_time)

    path, path_cost = rrtstar_planner.plan()

    plan_drawer_rrtstar = PlanDrawer(map_name, map_width, map_height, font_size)
    plan_drawer_rrtstar.draw_final(rrtstar_planner, path, path_cost)
    plan_drawer_rrtstar.animate_differential_drive_path(chronological_path(path, x_init), robot_radius, fps)

    # Keep the RRT* window open until Escape is pressed, instead of exiting immediately
    # and closing it as soon as the tree is drawn.
    while plan_drawer_rrtstar.stop_drawing_ == 0:
        plan_drawer_rrtstar.dispatch_events()


if __name__ == '__main__':

    main()
