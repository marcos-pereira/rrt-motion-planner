#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.

#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors:
# marcos-pereira (https://github.com/marcos-pereira)

from random_config import random
from Sampler import Sampler
from RealVectorState import RealVectorState

class DifferentialDriveSampler(Sampler):
    """ Sampler that draws uniform random control inputs (linear velocity, angular
    velocity) for a differential drive robot, bounded by [linear_velocity_min,
    linear_velocity_max] and [angular_velocity_min, angular_velocity_max].
    """

    def __init__(self, linear_velocity_min, linear_velocity_max,
                 angular_velocity_min, angular_velocity_max):
        """ Return a DifferentialDriveSampler object.

        Args:
            linear_velocity_min (float): the minimum linear velocity that can be sampled.
            linear_velocity_max (float): the maximum linear velocity that can be sampled.
            angular_velocity_min (float): the minimum angular velocity that can be sampled.
            angular_velocity_max (float): the maximum angular velocity that can be sampled.
        """
        self.linear_velocity_min_ = linear_velocity_min
        self.linear_velocity_max_ = linear_velocity_max
        self.angular_velocity_min_ = angular_velocity_min
        self.angular_velocity_max_ = angular_velocity_max

    def get_sample(self) -> RealVectorState:
        """ Return a uniformly sampled (linear_velocity, angular_velocity) control input
        within the bounds of this sampler.

        Returns:
            RealVectorState: the sampled (linear_velocity, angular_velocity) control input.
        """
        linear_velocity = random.uniform(self.linear_velocity_min_, self.linear_velocity_max_)
        angular_velocity = random.uniform(self.angular_velocity_min_, self.angular_velocity_max_)

        control_rand = RealVectorState((linear_velocity, angular_velocity))

        return control_rand
