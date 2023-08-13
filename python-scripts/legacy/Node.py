#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.
  
#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors: 
# marcos-pereira (https://github.com/marcos-pereira)

class Node:
    def __init__(self,
                 point,
                 parent,
                 index,
                 children,
                 cost):
        self.point_ = point
        self.parent_ = parent
        self.index_ = index
        self.children_ = children
        self.cost_ = cost

    def point(self):
        return self.point_

    def parent(self):
        return self.parent_

    def index(self):
        return self.index_

    def children(self):
        return self.children_

    def cost(self):
        return self.cost_
