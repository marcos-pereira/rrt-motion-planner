#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.

#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors:
# marcos-pereira (https://github.com/marcos-pereira)

import numpy as np


def get_angular_wheel_speeds_from_control_signal(control_signal: np.ndarray,
                                                   wheel_radius: float,
                                                   distance_wheels: float) -> tuple[float, float]:
    """ Convert a [linear_velocity, _, angular_velocity] control signal into the
    right and left wheel angular speeds of a differential drive robot.

    Args:
        control_signal (np.ndarray): 3-element vector with the linear velocity at
        index 0 and the angular velocity at index 2 (index 1 is unused, kept to match
        the control signal produced by a holonomic-base controller).
        wheel_radius (float): the radius of the wheels.
        distance_wheels (float): the distance between the two wheels.

    Returns:
        float: the right wheel angular speed.
        float: the left wheel angular speed.
    """
    if control_signal.size != 3:
        raise ValueError("Control signal must have 3 elements.")

    linear_velocity = control_signal[0]
    angular_velocity = control_signal[2]

    right_wheel_linear_velocity = linear_velocity + (angular_velocity * distance_wheels / 2.0)
    left_wheel_linear_velocity = linear_velocity - (angular_velocity * distance_wheels / 2.0)

    right_wheel_angular_speed = right_wheel_linear_velocity / wheel_radius
    left_wheel_angular_speed = left_wheel_linear_velocity / wheel_radius

    return right_wheel_angular_speed, left_wheel_angular_speed


def jacobian_differential_drive(state: np.ndarray, wheel_radius: float, distance_wheels: float) -> np.ndarray:
    """ Return the Jacobian matrix that maps the [right_wheel_angular_speed,
    left_wheel_angular_speed] vector to the [x_dot, y_dot, theta_dot] single
    integrator input of a differential drive robot.

    Args:
        state (np.ndarray): 3-element state vector [x, y, theta].
        wheel_radius (float): the radius of the wheels.
        distance_wheels (float): the distance between the two wheels.

    Returns:
        np.ndarray: the 3x2 Jacobian matrix.
    """
    if state.size != 3:
        raise ValueError("State vector must have 3 elements.")

    theta = state[2]

    jacobian = np.array([
        [(wheel_radius / 2.0) * np.cos(theta), (wheel_radius / 2.0) * np.cos(theta)],
        [(wheel_radius / 2.0) * np.sin(theta), (wheel_radius / 2.0) * np.sin(theta)],
        [(wheel_radius / distance_wheels), -(wheel_radius / distance_wheels)],
    ])

    return jacobian
