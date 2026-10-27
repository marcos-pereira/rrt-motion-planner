#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.

#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors:
# marcos-pereira (https://github.com/marcos-pereira)

#!/usr/bin/env python
""" Check that RRT* keeps its tree bookkeeping consistent through rewiring: node_to_parent_,
node_to_cost_, and the TreeNode parent/children/cost pointers must all describe the same tree.

Run with:
    python3 -m unittest test_rrtstar_tree_consistency
"""

import contextlib
import io
import math
import unittest

import numpy as np

from random_config import random
from benchmark_planners import build_planner, load_scene_map
from TreeNode import TreeNode
from RealVectorState import RealVectorState


class RRTStarTreeConsistencyTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scene_map_ = load_scene_map("smile.png")

    def check_tree_consistency(self, robot):
        """ Grow an RRT* tree and check every node's bookkeeping against its parent. """
        random.seed(2026)
        rrtstar = build_planner("rrtstar", robot, self.scene_map_, 3000)

        # The planner prints on every goal hit
        with contextlib.redirect_stdout(io.StringIO()):
            while rrtstar.node_count_ < 3000:
                rrtstar.run_step()

        root_node = rrtstar.tree_nodes_[0]
        children_count = 0

        for tree_node in rrtstar.tree_nodes_:
            node_coords = tree_node.get_node_coordinates()
            parent_node = tree_node.get_parent()
            children_count += len(tree_node.get_children())

            # The TreeNode cost must match node_to_cost_ exactly, including after rewiring
            self.assertEqual(tree_node.get_cost(), rrtstar.node_to_cost_[node_coords])

            if tree_node is root_node:
                self.assertIsNone(parent_node)
                continue

            parent_coords = parent_node.get_node_coordinates()
            self.assertEqual(rrtstar.node_to_parent_[node_coords], parent_coords)
            self.assertIn(tree_node, parent_node.get_children())
            self.assertAlmostEqual(rrtstar.node_to_cost_[node_coords],
                                   rrtstar.node_to_cost_[parent_coords] + rrtstar.nodes_distance(parent_coords, node_coords))

        # Every node except the root is the child of exactly one node
        self.assertEqual(children_count, len(rrtstar.tree_nodes_) - 1)

    def test_point_robot_tree_is_consistent(self):
        self.check_tree_consistency("point")

    def test_differential_drive_tree_is_consistent(self):
        self.check_tree_consistency("differential-drive")

    def test_update_child_costs_handles_deep_subtrees(self):
        """ A chain far deeper than Python's recursion limit must not raise RecursionError. """
        rrtstar = build_planner("rrtstar", "point", np.zeros((10, 10)), 10)
        chain_length = 5000

        root_node = TreeNode(RealVectorState((0, 0)), 0.0, None)
        rrtstar.node_to_cost_[(0, 0)] = 0.0
        parent_node = root_node
        for x in range(1, chain_length):
            child_node = TreeNode(RealVectorState((x, 0)), 0.0, parent_node)
            parent_node.add_child(child_node)
            rrtstar.node_to_cost_[(x, 0)] = 0.0
            parent_node = child_node

        rrtstar.update_child_costs(root_node)

        # Each unit step along the chain adds a cost of 1
        self.assertEqual(rrtstar.node_to_cost_[(chain_length - 1, 0)], chain_length - 1)
        self.assertEqual(parent_node.get_cost(), chain_length - 1)
        self.assertTrue(math.isclose(rrtstar.node_to_cost_[(2500, 0)], 2500))


if __name__ == "__main__":
    unittest.main()
