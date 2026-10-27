#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.

#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors:
# marcos-pereira (https://github.com/marcos-pereira)

#!/usr/bin/env python
""" Check NearestNeighborIndex against a brute-force search over the same nodes.

Run with:
    python3 -m unittest test_nearest_neighbor_index
"""

import math
import random
import unittest

from NearestNeighborIndex import NearestNeighborIndex


def brute_force_distance(node1, node2):
    """ Return the Euclidean distance between node1 and node2. """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(node1, node2)))


class NearestNeighborIndexTest(unittest.TestCase):

    def check_against_brute_force(self, nodes, queries, radius):
        """ Add nodes one by one and, after each addition, compare the index's answers for
        every query with a brute-force search over the nodes added so far. """
        nearest_neighbor_index = NearestNeighborIndex(nodes[0], initial_capacity=4, min_buffer_size=8)

        for node_count in range(1, len(nodes) + 1):
            if node_count > 1:
                nearest_neighbor_index.add(nodes[node_count - 1])
            nodes_added = nodes[:node_count]

            for query in queries:
                # Compare distances, not indexes, since equidistant nodes may be returned in either order
                nearest_index = nearest_neighbor_index.nearest(query)
                expected_distance = min(brute_force_distance(node, query) for node in nodes_added)
                self.assertAlmostEqual(brute_force_distance(nodes_added[nearest_index], query), expected_distance)

                expected_indexes = [i for i, node in enumerate(nodes_added)
                                    if brute_force_distance(node, query) <= radius]
                self.assertEqual(nearest_neighbor_index.within_radius(query, radius), expected_indexes)

    def test_integer_2d_nodes(self):
        generator = random.Random(0)
        nodes = [(generator.randint(0, 100), generator.randint(0, 100)) for _ in range(300)]
        queries = [(generator.randint(0, 100), generator.randint(0, 100)) for _ in range(10)]
        self.check_against_brute_force(nodes, queries, radius=12.5)

    def test_float_3d_nodes(self):
        generator = random.Random(1)
        nodes = [(generator.uniform(0, 100), generator.uniform(0, 100), generator.uniform(-math.pi, math.pi))
                 for _ in range(300)]
        queries = [(generator.uniform(0, 100), generator.uniform(0, 100), generator.uniform(-math.pi, math.pi))
                   for _ in range(10)]
        self.check_against_brute_force(nodes, queries, radius=15.0)

    def test_storage_grows_past_initial_capacity(self):
        nearest_neighbor_index = NearestNeighborIndex((0, 0), initial_capacity=2)
        for x in range(1, 50):
            nearest_neighbor_index.add((x, 0))
        self.assertEqual(nearest_neighbor_index.node_count_, 50)
        self.assertEqual(nearest_neighbor_index.nearest((48.9, 0.2)), 49)


if __name__ == "__main__":
    unittest.main()
