#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.

#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors:
# marcos-pereira (https://github.com/marcos-pereira)

import numpy as np
from Steer import Steer


class SimpleDeltaSteering(Steer):
    def steer(self, node1: tuple[int, int], node2: tuple[int, int], delta: float) -> tuple[int, int]:
        """ Returns a node between node1 and node2. If they are closer than delta, then
        return node2.

        Args:
            node1 (tuple): the initial node.
            node2 (tuple): the goal node towards which we steer.
            delta (double): the minimum distance to consider already near enough to node2.

        Returns:
            tuple: the new node between node1 and node2.
        """
        node1 = np.array([node1[0], node1[1]])
        node2 = np.array([node2[0], node2[1]])

        distance = np.linalg.norm(node2 - node1)

        if distance < delta:
            node = node2
        else:
            diffnodes = node2 - node1
            diffnodes = diffnodes / distance
            node = node1 + delta * diffnodes

        # Convert to int, otherwise the maps will not work with double precision
        # TODO: use some better mapping like a hash function to avoid this problem
        node = tuple(int(element) for element in node)

        return node
