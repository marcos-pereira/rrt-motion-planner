#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.

#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors:
# marcos-pereira (https://github.com/marcos-pereira)

from scipy.ndimage import distance_transform_edt

from CollisionChecker import CollisionChecker
from RealVectorState import RealVectorState

class DifferentialDriveCollisionChecker(CollisionChecker):
    """ Collision checker for a RealVectorState [x, y, theta] representing a
    DifferentialDriveRobot, checked against the obstacles of a scene map, where 0
    indicates free space and 1 indicates an obstacle. The robot is modeled as a
    circle of radius robot_radius_ centered at (x, y), and the check is done
    against a precomputed Euclidean Distance Transform (EDT) of the scene map, so
    that each collision query costs O(1).
    """

    def __init__(self, scene_map, robot_radius, cell_size=1.0):
        """ Return a DifferentialDriveCollisionChecker object.

        Args:
            scene_map (numpy matrix): the scene map where 0 indicate free space and 1
            indicate obstacles.
            robot_radius (float): the radius of the circle that models the footprint
            of the DifferentialDriveRobot.
            cell_size (float): the size, in world units, of one scene_map cell.
            Defaults to 1.0, i.e. world units and grid cells coincide.
        """
        self.robot_radius_ = robot_radius
        self.cell_size_ = cell_size

        # Distance, in cells, from each free cell to the nearest obstacle cell.
        # Obstacle cells themselves get a distance of 0.
        self.distance_field_ = distance_transform_edt(scene_map == 0)

        self.map_height_, self.map_width_ = scene_map.shape

    def collision(self, state: RealVectorState) -> bool:
        """ Return if the (x, y) position of state is in collision with the
        obstacles of the scene map. A position outside the bounds of the scene map
        is considered free, consistent with a scene map bounded to the planning
        space.

        Args:
            state (RealVectorState): the [x, y, theta] state to check for collision.

        Returns:
            bool: True if state is in collision, false otherwise.
        """
        x, y, _ = state.get_value()

        col = int(x // self.cell_size_)
        row = int(y // self.cell_size_)

        if col < 0 or col >= self.map_width_ or row < 0 or row >= self.map_height_:
            return False

        distance_to_obstacle = self.distance_field_[row, col] * self.cell_size_

        return distance_to_obstacle <= self.robot_radius_
