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

class RealVectorState(State):
    """ State representation of a point in a real vector space R^n, e.g. R2, R3,
    and so on. The state is stored as a tuple of real numbers, one per dimension.
    """

    def __init__(self, value: tuple):
        """ Return a RealVectorState object.

        Args:
            value (tuple): the coordinates of the point in R^n that represents this state.
        """
        self.set_value(value)

    def get_value(self) -> tuple:
        """ Return the coordinates of this state.

        Returns:
            tuple: the coordinates of the point in R^n that represents this state.
        """
        return self.value_

    def set_value(self, value: tuple):
        """ Set the coordinates of this state.

        Args:
            value (tuple): the new coordinates of the point in R^n that represents this state.
        """
        self.value_ = tuple(value)

    def dimension(self) -> int:
        """ Return the dimension n of the R^n space this state belongs to.

        Returns:
            int: the number of coordinates of this state.
        """
        return len(self.value_)

    def __eq__(self, other):
        if isinstance(other, RealVectorState):
            return self.value_ == other.value_
        return NotImplemented

    def __hash__(self):
        return hash(self.value_)

    def __repr__(self):
        return f"RealVectorState{self.value_}"
