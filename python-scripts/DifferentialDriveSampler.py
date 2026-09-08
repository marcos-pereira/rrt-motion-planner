#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.

#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors:
# marcos-pereira (https://github.com/marcos-pereira)

import math

import numpy as np

from DifferentialDriveRobot import DifferentialDriveRobot
from random_config import random
from Sampler import Sampler
from RealVectorState import RealVectorState

class DifferentialDriveSampler(Sampler):
    """ Sampler that draws a random [x, y, theta] pose reached by a DifferentialDriveRobot
    after applying a randomly sampled [linear_velocity, angular_velocity] control for one
    sampling step, starting from a random base pose drawn uniformly across the given map
    bounds. Unlike sampling a pose directly (e.g. DifferentialDrivePoseSampler), the
    returned pose is computed by actually simulating the robot's unicycle dynamics, so it
    reflects a state genuinely reachable under the robot's motion model, while still
    covering the whole map through the randomized base pose.
    """

    def __init__(self,
                 linear_velocity_min, linear_velocity_max,
                 angular_velocity_min, angular_velocity_max,
                 x_min, x_max, y_min, y_max,
                 wheel_radius, distance_wheels, sampling_time,
                 theta_min=-math.pi, theta_max=math.pi):
        """ Return a DifferentialDriveSampler object.

        Args:
            linear_velocity_min (float): the minimum linear velocity that can be sampled.
            linear_velocity_max (float): the maximum linear velocity that can be sampled.
            angular_velocity_min (float): the minimum angular velocity that can be sampled.
            angular_velocity_max (float): the maximum angular velocity that can be sampled.
            x_min (float): the minimum x coordinate of the base pose that can be sampled.
            x_max (float): the maximum x coordinate of the base pose that can be sampled.
            y_min (float): the minimum y coordinate of the base pose that can be sampled.
            y_max (float): the maximum y coordinate of the base pose that can be sampled.
            wheel_radius (float): the radius of the wheels of the simulated robot.
            distance_wheels (float): the distance between the two wheels of the
            simulated robot.
            sampling_time (float): the duration, in seconds, of the simulated step.
            theta_min (float): the minimum heading of the base pose that can be sampled.
            Defaults to -pi.
            theta_max (float): the maximum heading of the base pose that can be sampled.
            Defaults to pi.
        """
        self.linear_velocity_min_ = linear_velocity_min
        self.linear_velocity_max_ = linear_velocity_max
        self.angular_velocity_min_ = angular_velocity_min
        self.angular_velocity_max_ = angular_velocity_max
        self.x_min_ = x_min
        self.x_max_ = x_max
        self.y_min_ = y_min
        self.y_max_ = y_max
        self.theta_min_ = theta_min
        self.theta_max_ = theta_max
        self.wheel_radius_ = wheel_radius
        self.distance_wheels_ = distance_wheels
        self.sampling_time_ = sampling_time

    def get_sample(self) -> RealVectorState:
        """ Return the [x, y, theta] pose reached by a DifferentialDriveRobot, initialized
        at a random base pose within this sampler's bounds, after applying a randomly
        sampled [linear_velocity, angular_velocity] control for one sampling step.

        Returns:
            RealVectorState: the sampled [x, y, theta] pose.
        """
        base_x = random.uniform(self.x_min_, self.x_max_)
        base_y = random.uniform(self.y_min_, self.y_max_)
        base_theta = random.uniform(self.theta_min_, self.theta_max_)

        linear_velocity = random.uniform(self.linear_velocity_min_, self.linear_velocity_max_)
        angular_velocity = random.uniform(self.angular_velocity_min_, self.angular_velocity_max_)
        control_input = np.array([linear_velocity, 0.0, angular_velocity])

        robot = DifferentialDriveRobot(np.array([base_x, base_y, base_theta]), self.sampling_time_,
                                        self.wheel_radius_, self.distance_wheels_)

        pose_rand = RealVectorState(tuple(robot.get_state(control_input)))

        return pose_rand
