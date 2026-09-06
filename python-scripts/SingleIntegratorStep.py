#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.

#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors:
# marcos-pereira (https://github.com/marcos-pereira)

import numpy as np
from Step import Step


class SingleIntegratorStep(Step):
    def step(self, state: np.ndarray, control_input: np.ndarray, sampling_time: float) -> np.ndarray:
        """ Return the next state using single integration, i.e. state + control_input * sampling_time.

        Args:
            state (np.ndarray): the current state vector.
            control_input (np.ndarray): the control input vector applied to the model.
            sampling_time (float): the duration, in seconds, over which the input is applied.

        Returns:
            np.ndarray: the next state vector.
        """
        return state + control_input * sampling_time
