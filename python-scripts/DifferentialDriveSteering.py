#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.

#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors:
# marcos-pereira (https://github.com/marcos-pereira)

import numpy as np
from DifferentialDriveRobot import DifferentialDriveRobot
from SingleIntegratorStep import SingleIntegratorStep
from Steer import Steer


class DifferentialDriveSteering(Steer):
    def __init__(self,
                 wheel_radius: float,
                 distance_wheels: float,
                 sampling_time: float):
        """ Return a DifferentialDriveSteering object that steers between states
        [x, y, theta] by simulating a DifferentialDriveRobot for one sampling step
        towards node2.

        Args:
            wheel_radius (float): the radius of the wheels of the simulated robot.
            distance_wheels (float): the distance between the two wheels of the
            simulated robot.
            sampling_time (float): the duration, in seconds, of each simulated step.
        """
        self.wheel_radius_ = wheel_radius
        self.distance_wheels_ = distance_wheels
        self.sampling_time_ = sampling_time

    def steer(self,
              node1: tuple[float, float, float],
              node2: tuple[float, float, float],
              delta: float) -> tuple[float, float, float]:
        """ Return the state reached by driving a DifferentialDriveRobot, initialized
        at node1, towards node2. The robot first rotates in place to align its heading
        with node2, then drives straight towards node2 with linear velocity, each as a
        separate sampling step of the same robot. If node1 and node2 are already closer
        than delta, node2 is returned directly.

        Args:
            node1 (tuple): the state [x, y, theta] from which we steer.
            node2 (tuple): the state [x, y, theta] towards which we steer.
            delta (double): the step size (or the minimum distance to node2 to consider
            it already reached).

        Returns:
            tuple: the state [x, y, theta] reached by steering from node1 towards node2.
        """
        node1_array = np.array(node1, dtype=float)
        node2_array = np.array(node2, dtype=float)

        distance = np.linalg.norm(node2_array[:2] - node1_array[:2])
        if distance < delta:
            return tuple(node2_array)

        heading_to_goal = np.arctan2(node2_array[1] - node1_array[1], node2_array[0] - node1_array[0])
        heading_error = np.arctan2(np.sin(heading_to_goal - node1_array[2]), np.cos(heading_to_goal - node1_array[2]))

        robot = DifferentialDriveRobot(node1_array, self.sampling_time_, self.wheel_radius_,
                                        self.distance_wheels_, SingleIntegratorStep())

        # Rotate in place to align the heading with node2.
        align_control_input = np.array([0.0, 0.0, heading_error / self.sampling_time_])
        robot.get_state(align_control_input)

        # Drive straight towards node2 at a linear speed that covers delta in
        # one sampling step.
        drive_control_input = np.array([delta / self.sampling_time_, 0.0, 0.0])
        node = robot.get_state(drive_control_input)

        return tuple(node)
