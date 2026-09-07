#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.

#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors:
# marcos-pereira (https://github.com/marcos-pereira)

from abc import ABC, abstractmethod


class Sampler(ABC):
    @abstractmethod
    def get_sample(self) -> tuple:
        """ Return a sampled configuration from the space this sampler draws from.

        Returns:
            tuple: the sampled configuration.
        """
        pass
