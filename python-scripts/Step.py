#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.

#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors:
# marcos-pereira (https://github.com/marcos-pereira)

from abc import ABC, abstractmethod

import numpy as np


class Step(ABC):
    @abstractmethod
    def step(self, state: np.ndarray, control_input: np.ndarray, sampling_time: float) -> np.ndarray:
        """ Return the next state obtained by integrating control_input from state
        over sampling_time. Concrete implementations decide the integration model used.

        Args:
            state (np.ndarray): the current state vector.
            control_input (np.ndarray): the control input vector applied to the model.
            sampling_time (float): the duration, in seconds, over which the input is applied.

        Returns:
            np.ndarray: the next state vector.
        """
        pass
