#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.

#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors:
# marcos-pereira (https://github.com/marcos-pereira)

from abc import ABC, abstractmethod


class Steer(ABC):
    @abstractmethod
    def steer(self, node1: tuple[int, int], node2: tuple[int, int], delta: float) -> tuple[int, int]:
        """ Return the node to move to when expanding the tree from node1 towards node2.
        Concrete implementations decide how the step defined by delta is taken.

        Args:
            node1 (tuple): the node in the tree from which we steer.
            node2 (tuple): the node towards which we steer.
            delta (double): the step size (or the minimum distance to node2 to consider
            it already reached) used by the concrete steering strategy.

        Returns:
            tuple: the new node produced by steering from node1 towards node2.
        """
        pass
