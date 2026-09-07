#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.

#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors:
# marcos-pereira (https://github.com/marcos-pereira)

from random_config import random
from Sampler import Sampler

class Cartesian2DSampler(Sampler):
    """ Sampler that draws uniform random points in a 2D Cartesian space bounded
    by [x_min, x_max] and [y_min, y_max].
    """

    def __init__(self, x_min, x_max, y_min, y_max):
        """ Return a Cartesian2DSampler object.

        Args:
            x_min (int): the minimum x coordinate that can be sampled.
            x_max (int): the maximum x coordinate that can be sampled.
            y_min (int): the minimum y coordinate that can be sampled.
            y_max (int): the maximum y coordinate that can be sampled.
        """
        self.x_min_ = x_min
        self.x_max_ = x_max
        self.y_min_ = y_min
        self.y_max_ = y_max

    def get_sample(self) -> tuple:
        """ Return a uniformly sampled (x, y) point within the bounds of this sampler.

        Returns:
            tuple: the sampled (x, y) configuration.
        """
        x = random.randint(self.x_min_, self.x_max_)
        y = random.randint(self.y_min_, self.y_max_)

        x_rand = (x, y)

        return x_rand
