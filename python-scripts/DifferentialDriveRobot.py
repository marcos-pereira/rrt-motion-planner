#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.

#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors:
# marcos-pereira (https://github.com/marcos-pereira)

import numpy as np
from differential_drive_kinematics import (get_angular_wheel_speeds_from_control_signal,
                                            jacobian_differential_drive)
from Robot import Robot
from SingleIntegratorStep import SingleIntegratorStep
from Step import Step


class DifferentialDriveRobot(Robot):
    def __init__(self,
                 initial_state: np.ndarray,
                 sampling_time: float,
                 wheel_radius: float,
                 distance_wheels: float,
                 step: Step = None):
        """ Return a DifferentialDriveRobot object that simulates a differential
        drive robot with state [x, y, theta].

        Args:
            initial_state (np.ndarray): the initial state vector [x, y, theta].
            sampling_time (float): the duration, in seconds, of each simulation step.
            wheel_radius (float): the radius of the wheels.
            distance_wheels (float): the distance between the two wheels.
            step (Step): the integration strategy used to update the robot state.
            Defaults to SingleIntegratorStep, as a differential drive robot is a
            single integrator in the wheel angular speeds' Jacobian-mapped input.
        """
        super().__init__(initial_state, sampling_time, step or SingleIntegratorStep())
        self.wheel_radius_ = wheel_radius
        self.distance_wheels_ = distance_wheels

    def get_state(self, control_input: np.ndarray) -> np.ndarray:
        """ Compute and return the next robot state resulting from applying a
        [linear_velocity, _, angular_velocity] control input to the current state.
        The control input is converted into wheel angular speeds, mapped back into
        a single integrator input through the differential drive Jacobian, and then
        integrated by the Step strategy.

        Args:
            control_input (np.ndarray): 3-element vector with the linear velocity at
            index 0 and the angular velocity at index 2.

        Returns:
            np.ndarray: the next state vector [x, y, theta].
        """
        right_wheel_speed, left_wheel_speed = get_angular_wheel_speeds_from_control_signal(
            control_input, self.wheel_radius_, self.distance_wheels_)
        wheel_speeds = np.array([right_wheel_speed, left_wheel_speed])

        jacobian = jacobian_differential_drive(self.state_, self.wheel_radius_, self.distance_wheels_)
        single_integrator_input = jacobian @ wheel_speeds

        self.state_ = self.step_.step(self.state_, single_integrator_input, self.sampling_time_)

        return self.state_
