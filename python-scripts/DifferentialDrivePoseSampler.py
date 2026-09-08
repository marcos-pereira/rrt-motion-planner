#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.

#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors:
# marcos-pereira (https://github.com/marcos-pereira)

import math

from random_config import random
from Sampler import Sampler
from RealVectorState import RealVectorState

class DifferentialDrivePoseSampler(Sampler):
    """ Sampler that draws uniform random poses [x, y, theta] for a differential drive
    robot, with (x, y) bounded by [x_min, x_max] and [y_min, y_max], and theta bounded
    by [theta_min, theta_max]. This is the sampler_ passed to RRT/RRTStar: the sampled
    pose is used for the nearest-neighbor bias towards unexplored regions of the state
    space, independently of the control sampling done by the Steer strategy (e.g.
    DifferentialDriveRandomControlSteering).
    """

    def __init__(self, x_min, x_max, y_min, y_max, theta_min=-math.pi, theta_max=math.pi):
        """ Return a DifferentialDrivePoseSampler object.

        Args:
            x_min (float): the minimum x coordinate that can be sampled.
            x_max (float): the maximum x coordinate that can be sampled.
            y_min (float): the minimum y coordinate that can be sampled.
            y_max (float): the maximum y coordinate that can be sampled.
            theta_min (float): the minimum heading that can be sampled. Defaults to -pi.
            theta_max (float): the maximum heading that can be sampled. Defaults to pi.
        """
        self.x_min_ = x_min
        self.x_max_ = x_max
        self.y_min_ = y_min
        self.y_max_ = y_max
        self.theta_min_ = theta_min
        self.theta_max_ = theta_max

    def get_sample(self) -> RealVectorState:
        """ Return a uniformly sampled [x, y, theta] pose within the bounds of this sampler.

        Returns:
            RealVectorState: the sampled [x, y, theta] configuration.
        """
        x = random.uniform(self.x_min_, self.x_max_)
        y = random.uniform(self.y_min_, self.y_max_)
        theta = random.uniform(self.theta_min_, self.theta_max_)

        pose_rand = RealVectorState((x, y, theta))

        return pose_rand
