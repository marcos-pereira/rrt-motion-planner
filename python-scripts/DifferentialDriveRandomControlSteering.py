#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.

#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors:
# marcos-pereira (https://github.com/marcos-pereira)

import numpy as np

from DifferentialDriveRobot import DifferentialDriveRobot
from Sampler import Sampler
from Steer import Steer


class DifferentialDriveRandomControlSteering(Steer):
    """ Steering strategy that expands the tree with a randomly sampled control input,
    following the kinodynamic RRT formulation in S. LaValle, "Planning Algorithms",
    Section 5.3.1: rather than steering directly towards the sampled configuration, a
    control is drawn from control_sampler and applied to a DifferentialDriveRobot
    initialized at node1 for one sampling step, and the resulting state becomes the new
    tree node. node2, the configuration sampled to bias the search, is only used
    upstream to pick node1 as its nearest neighbor in the tree; it plays no role here.
    """

    def __init__(self,
                 control_sampler: Sampler,
                 wheel_radius: float,
                 distance_wheels: float,
                 sampling_time: float):
        """ Return a DifferentialDriveRandomControlSteering object.

        Args:
            control_sampler (Sampler): the sampling strategy used to draw random
            [linear_velocity, angular_velocity] control inputs, e.g. DifferentialDriveSampler.
            wheel_radius (float): the radius of the wheels of the simulated robot.
            distance_wheels (float): the distance between the two wheels of the
            simulated robot.
            sampling_time (float): the duration, in seconds, of the simulated step.
        """
        self.control_sampler_ = control_sampler
        self.wheel_radius_ = wheel_radius
        self.distance_wheels_ = distance_wheels
        self.sampling_time_ = sampling_time

    def steer(self,
              node1: tuple[float, float, float],
              node2: tuple[float, float, float],
              delta: float) -> tuple[float, float, float]:
        """ Return the state reached by applying a randomly sampled control input to a
        DifferentialDriveRobot initialized at node1, for one sampling step. node2 and
        delta are unused, since the new tree node is determined by the sampled control
        rather than by steering towards a target configuration.

        Args:
            node1 (tuple): the state [x, y, theta] from which we steer.
            node2 (tuple): unused; kept to satisfy the Steer interface.
            delta (double): unused; kept to satisfy the Steer interface.

        Returns:
            tuple: the state [x, y, theta] reached by applying the sampled control to node1.
        """
        linear_velocity, angular_velocity = self.control_sampler_.get_sample().get_value()
        control_input = np.array([linear_velocity, 0.0, angular_velocity])

        robot = DifferentialDriveRobot(np.array(node1, dtype=float), self.sampling_time_,
                                        self.wheel_radius_, self.distance_wheels_)

        return tuple(robot.get_state(control_input))
