#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.

#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors:
# marcos-pereira (https://github.com/marcos-pereira)

import math

import numpy as np
from scipy.spatial import cKDTree


class NearestNeighborIndex:
    """ Incremental nearest-neighbor index over the nodes of a growing tree, used by RRT and
    RRT* to answer nearest-node and radius queries without rebuilding a spatial index on every
    step.

    Nodes are stored in insertion order, so the index i returned by a query refers to the i-th
    node added. The nodes are split in two parts:
    - the nodes added up to the last rebuild, searched with a kd-tree in O(log n);
    - the nodes added since then (the buffer), searched with numpy brute force.
    The kd-tree is rebuilt once the buffer holds more than about sqrt(n) nodes, so both the
    amortized rebuild cost and the brute-force search stay small as the tree grows.
    """

    def __init__(self, first_node: tuple, initial_capacity=1024, min_buffer_size=64):
        """ Return a NearestNeighborIndex object holding first_node.

        Args:
            first_node (tuple): the first node of the tree, e.g. the initial configuration.
            Its length sets the dimension of every node in the index.
            initial_capacity (int): the number of nodes preallocated; the storage doubles
            whenever it is full.
            min_buffer_size (int): the buffer size below which the kd-tree is never rebuilt,
            since brute force over a few nodes is cheaper than a rebuild.
        """
        dimension = len(first_node)

        # Coordinates of all nodes, in insertion order. Only the first node_count_ rows are used.
        self.nodes_ = np.empty((initial_capacity, dimension))
        self.node_count_ = 0

        # kd-tree over the first kd_tree_node_count_ nodes, or None before the first rebuild
        self.kd_tree_ = None
        self.kd_tree_node_count_ = 0

        self.min_buffer_size_ = min_buffer_size

        self.add(first_node)

    def add(self, node: tuple):
        """ Add node to the index.

        Args:
            node (tuple): the node to add, with the same dimension as the first node.
        """
        # Double the storage when it is full
        if self.node_count_ == len(self.nodes_):
            self.nodes_ = np.concatenate([self.nodes_, np.empty_like(self.nodes_)])

        self.nodes_[self.node_count_] = node
        self.node_count_ += 1

        buffer_size = self.node_count_ - self.kd_tree_node_count_
        if buffer_size > max(self.min_buffer_size_, math.sqrt(self.node_count_)):
            self.rebuild_kd_tree()

    def rebuild_kd_tree(self):
        """ Rebuild the kd-tree over all nodes added so far, emptying the buffer. """
        # Unbalanced, non-compact trees build several times faster and query about as fast
        self.kd_tree_ = cKDTree(self.nodes_[:self.node_count_], balanced_tree=False, compact_nodes=False)
        self.kd_tree_node_count_ = self.node_count_

    def nearest(self, query: tuple) -> int:
        """ Return the index of the node nearest to query, by Euclidean distance.

        Args:
            query (tuple): the point to search from, with the same dimension as the nodes.

        Returns:
            int: the insertion index of the nearest node.
        """
        query_point = np.asarray(query, dtype=float)

        nearest_index = -1
        nearest_distance_squared = math.inf

        if self.kd_tree_ is not None:
            kd_tree_distance, kd_tree_index = self.kd_tree_.query(query_point, k=1)
            nearest_index = int(kd_tree_index)
            nearest_distance_squared = kd_tree_distance * kd_tree_distance

        buffer = self.nodes_[self.kd_tree_node_count_:self.node_count_]
        if len(buffer) > 0:
            buffer_distances_squared = ((buffer - query_point) ** 2).sum(axis=1)
            buffer_index = int(np.argmin(buffer_distances_squared))

            # Strict comparison keeps the older node on ties
            if buffer_distances_squared[buffer_index] < nearest_distance_squared:
                nearest_index = self.kd_tree_node_count_ + buffer_index

        return nearest_index

    def within_radius(self, query: tuple, radius: float) -> list[int]:
        """ Return the indexes of all nodes within radius of query, by Euclidean distance.

        Args:
            query (tuple): the point to search from, with the same dimension as the nodes.
            radius (float): the radius of the ball around query, with its boundary included.

        Returns:
            list: the insertion indexes of the nodes within radius, in increasing order.
        """
        query_point = np.asarray(query, dtype=float)

        indexes = []

        if self.kd_tree_ is not None:
            indexes = self.kd_tree_.query_ball_point(query_point, radius, return_sorted=True)

        buffer = self.nodes_[self.kd_tree_node_count_:self.node_count_]
        if len(buffer) > 0:
            buffer_distances_squared = ((buffer - query_point) ** 2).sum(axis=1)
            buffer_indexes = np.flatnonzero(buffer_distances_squared <= radius * radius)
            indexes = list(indexes) + (buffer_indexes + self.kd_tree_node_count_).tolist()

        return indexes
