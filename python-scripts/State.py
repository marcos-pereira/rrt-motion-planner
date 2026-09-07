#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.

#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors:
# marcos-pereira (https://github.com/marcos-pereira)

from abc import ABC, abstractmethod

class State(ABC):
    """ Abstract representation of a system state (a configuration) handled by the
    RRT-based planners, such as x_init, x_goal or any node sampled or added to the
    tree while planning. Concrete subclasses define the underlying representation
    of the state.
    """

    @abstractmethod
    def get_value(self):
        """ Return the value that represents this state.
        """
        pass

    @abstractmethod
    def set_value(self, value):
        """ Set the value that represents this state.

        Args:
            value: the new value that represents this state.
        """
        pass
