#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.

#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors:
# marcos-pereira (https://github.com/marcos-pereira)

from abc import ABC, abstractmethod

from State import State

class CollisionChecker(ABC):
    """ Abstract collision checking strategy used by RRT-based planners to check if a
    state is in collision with the obstacles of the state space. Concrete subclasses
    define the state space and the obstacle representation the check is performed against.
    """

    @abstractmethod
    def collision(self, state: State) -> bool:
        """ Return if state is in collision with the obstacles of the state space.

        Args:
            state (State): the state to check for collision.

        Returns:
            bool: True if state is in collision, false otherwise.
        """
        pass
