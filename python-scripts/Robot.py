#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.

#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors:
# marcos-pereira (https://github.com/marcos-pereira)

from abc import ABC, abstractmethod

import numpy as np
from Step import Step


class Robot(ABC):
    def __init__(self, initial_state: np.ndarray, sampling_time: float, step: Step):
        """ Return a Robot object that simulates a robot model by integrating a
        control input vector into its state using a pluggable Step strategy.

        Args:
            initial_state (np.ndarray): the initial state vector of the robot.
            sampling_time (float): the duration, in seconds, of each simulation step.
            step (Step): the integration strategy used to update the robot state.
        """
        self.state_ = initial_state
        self.sampling_time_ = sampling_time
        self.step_ = step

    @abstractmethod
    def get_state(self, control_input: np.ndarray) -> np.ndarray:
        """ Compute and return the next robot state resulting from applying
        control_input to the current state. Concrete implementations decide how
        the control input is converted into the model used by the Step strategy.

        Args:
            control_input (np.ndarray): the control input vector applied to the robot.

        Returns:
            np.ndarray: the next state vector.
        """
        pass
