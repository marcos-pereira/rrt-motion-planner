#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.

#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors:
# marcos-pereira (https://github.com/marcos-pereira)

import math

from Steer import Steer


class DifferentialDriveSteering(Steer):
    def __init__(self,
                 wheel_radius: float,
                 distance_wheels: float,
                 sampling_time: float):
        """ Return a DifferentialDriveSteering object that steers between states
        [x, y, theta] following a DifferentialDriveRobot model towards node2. The
        robot parameters define that model, but they cancel out in the closed form
        used by steer(), so the steered state does not depend on them.

        Args:
            wheel_radius (float): the radius of the wheels of the modeled robot.
            distance_wheels (float): the distance between the two wheels of the
            modeled robot.
            sampling_time (float): the duration, in seconds, of each modeled step.
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
        x1, y1, theta1 = (float(element) for element in node1)
        x2, y2 = float(node2[0]), float(node2[1])

        distance = math.hypot(x2 - x1, y2 - y1)
        if distance < delta:
            return tuple(float(element) for element in node2)

        heading_to_goal = math.atan2(y2 - y1, x2 - x1)
        heading_error = math.atan2(math.sin(heading_to_goal - theta1), math.cos(heading_to_goal - theta1))

        # The two DifferentialDriveRobot sampling steps above have a closed form, used here
        # instead of simulating the robot because steer() runs on every sampling attempt.
        # With a single integrator, rotating in place at angular velocity heading_error / T
        # for one step T only changes the heading, and driving straight at linear velocity
        # delta / T for one step moves delta along the new heading (the wheel radius and
        # distance cancel out in the Jacobian). test_differential_drive_steering.py checks
        # this against the simulated robot.
        theta_new = theta1 + heading_error
        x_new = x1 + delta * math.cos(theta_new)
        y_new = y1 + delta * math.sin(theta_new)

        return (x_new, y_new, theta_new)
