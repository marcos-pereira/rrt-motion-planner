#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.

#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors:
# marcos-pereira (https://github.com/marcos-pereira)

#!/usr/bin/env python
""" Check that DifferentialDriveSteering's closed form matches simulating the two
DifferentialDriveRobot sampling steps it replaces: rotate in place towards node2, then
drive straight for delta.

Run with:
    python3 -m unittest test_differential_drive_steering
"""

import math
import random
import unittest

import numpy as np

from DifferentialDriveRobot import DifferentialDriveRobot
from DifferentialDriveSteering import DifferentialDriveSteering
from SingleIntegratorStep import SingleIntegratorStep


def simulate_steer(node1, node2, delta, wheel_radius, distance_wheels, sampling_time):
    """ Return the state reached by simulating a DifferentialDriveRobot that first rotates
    in place to face node2 and then drives straight for delta, one sampling step each. """
    heading_to_goal = math.atan2(node2[1] - node1[1], node2[0] - node1[0])
    heading_error = math.atan2(math.sin(heading_to_goal - node1[2]), math.cos(heading_to_goal - node1[2]))

    robot = DifferentialDriveRobot(np.array(node1, dtype=float), sampling_time, wheel_radius,
                                   distance_wheels, SingleIntegratorStep())
    robot.get_state(np.array([0.0, 0.0, heading_error / sampling_time]))
    node = robot.get_state(np.array([delta / sampling_time, 0.0, 0.0]))

    return tuple(node)


class DifferentialDriveSteeringTest(unittest.TestCase):

    def test_matches_simulated_robot(self):
        generator = random.Random(0)

        for _ in range(1000):
            wheel_radius = generator.uniform(0.1, 5.0)
            distance_wheels = generator.uniform(0.1, 5.0)
            sampling_time = generator.uniform(0.1, 50.0)
            delta = generator.uniform(1.0, 30.0)
            node1 = (generator.uniform(0, 800), generator.uniform(0, 600), generator.uniform(-math.pi, math.pi))
            node2 = (generator.uniform(0, 800), generator.uniform(0, 600), generator.uniform(-math.pi, math.pi))

            # Only the far case simulates the robot; closer nodes return node2 directly
            if math.hypot(node2[0] - node1[0], node2[1] - node1[1]) < delta:
                continue

            steer = DifferentialDriveSteering(wheel_radius, distance_wheels, sampling_time)
            steered_node = steer.steer(node1, node2, delta)
            simulated_node = simulate_steer(node1, node2, delta, wheel_radius, distance_wheels, sampling_time)

            for steered_value, simulated_value in zip(steered_node, simulated_node):
                self.assertAlmostEqual(steered_value, simulated_value, places=9)

    def test_returns_node2_when_within_delta(self):
        steer = DifferentialDriveSteering(1.0, 1.0, 20.0)
        node2 = (105.0, 102.0, 0.5)
        self.assertEqual(steer.steer((100.0, 100.0, 0.0), node2, 20.0), node2)


if __name__ == "__main__":
    unittest.main()
