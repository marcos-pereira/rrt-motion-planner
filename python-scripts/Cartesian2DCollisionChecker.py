#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.

#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors:
# marcos-pereira (https://github.com/marcos-pereira)

import numpy as np

from CollisionChecker import CollisionChecker
from RealVectorState import RealVectorState

class Cartesian2DCollisionChecker(CollisionChecker):
    """ Collision checker for a RealVectorState in a 2D Cartesian space, checked against
    the obstacles of a scene map, where 0 indicates free space and 1 indicates an obstacle.
    """

    def __init__(self, scene_map):
        """ Return a Cartesian2DCollisionChecker object.

        Args:
            scene_map (numpy matrix): the scene map where 0 indicate free space and 1
            indicate obstacles.
        """
        # The rows and columns where the scene map has an obstacle
        ones_in_drawing = np.where(scene_map == 1)

        # The (x, y) coordinates of the obstacles in the scene map
        self.obstacles_coordinates_ = set(zip(ones_in_drawing[1], ones_in_drawing[0]))

    def collision(self, state: RealVectorState) -> bool:
        """ Return if state is in collision with the obstacles of the scene map. The state
        is converted to integers because the scene map is discretized into integers.

        Args:
            state (RealVectorState): the state to check for collision.

        Returns:
            bool: True if state is in collision, false otherwise.
        """
        state_integers = tuple(int(element) for element in state.get_value())

        return state_integers in self.obstacles_coordinates_
