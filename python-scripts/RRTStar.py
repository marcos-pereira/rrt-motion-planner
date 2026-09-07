#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.
  
#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors: 
# marcos-pereira (https://github.com/marcos-pereira)

import copy

import numpy as np
from RRTPlanner import RRTPlanner
from RealVectorState import RealVectorState
from Sampler import Sampler
from State import State
from Steer import Steer
from sklearn.neighbors import NearestNeighbors
from TreeNode import TreeNode

class RRTStar(RRTPlanner):
    def __init__(self,
                 state_init: State,
                 state_goal: State,
                 goal_radius,
                 steer_delta,
                 steer: Steer,
                 sampler: Sampler,
                 nearest_neighbor_eta,
                 gamma_rrt,
                 nearest_neighbor_radius,
                 scene_map,
                 max_num_nodes,
                 max_planning_time=None):
        """ Return RRTStar planner.

        Args:
            state_init (State): The initial configuration.
            state_goal (State): The goal configuration.
            goal_radius (_type_): Radius to be considered within the goal.
            steer_delta (_type_): Value used to steer toward the sampled configurations.
            steer (Steer): the steering strategy used to move from a node in the tree
            towards the sampled configurations.
            sampler (Sampler): the sampling strategy used to draw random configurations
            from the configuration space.
            nearest_neighbor_eta (double) : Gain used to determine radius of ball for nearest neighbors.
            gamma_rrt (_type_): Gain used to determine radius of ball for nearest neighbors.
            nearest_neighbor_radius (double): this parameter is not being used and will not take effect.
            scene_map (numpy matrix): Map of the scene or configuration space where 0 indicate free space and 1 indicate obstacle.
            max_num_nodes (_type_): Maximum number of nodes in the tree.
            max_planning_time (float): the maximum time in seconds that plan() may run, or
            None to only bound the search by max_num_nodes.
        """
        super().__init__(state_init,
                         state_goal,
                         goal_radius,
                         steer_delta,
                         steer,
                         sampler,
                         scene_map,
                         max_num_nodes,
                         max_planning_time)
        
        self.gamma_rrt_ = gamma_rrt
        self.nearest_neighbor_eta_ = nearest_neighbor_eta
        self.nearest_neighbor_radius_ = nearest_neighbor_radius
        
        # Initial cost to goal
        self.cost_to_goal_ = np.inf
        
        # Last cost to goal calculated
        self.last_cost_to_goal_ = np.inf
        
        ## Store last path found
        self.last_path_found_ = list()

        ## Store last node in goal
        self.last_goal_node_ = None

        # Node with minimum cost in tree to connect new node to
        self.state_min_ = RealVectorState(tuple())
        
        # If at least one path to goal found
        self.one_path_found_ = False
        
    def plan_found(self) -> tuple[bool, State, State]:
        """ Returns if a plan could be found, the nearest node to the reached node, and the reached node in goal radius.

        Returns:
            bool: True if a plan is found, false otherwise.
            State: the nearest node to the new node added.
            State: the new node found.
        """

        while True:
            state_rand = self.configuration_in_free_space()

            ## Get nearest node to state_rand
            state_nearest = self.nearest_node(state_rand, self.rrt_graph_)

            ## Steer from nearest node in tree (i.e. parent_node) towards the
            ## state_rand to obtain a new node for the tree
            state_new_value = self.steer_.steer(state_nearest.get_value(), state_rand.get_value(), self.steer_delta_)

            ## Check if node is in collision
            if self.collision(state_new_value) == True:
                # Node in collision
                # print("collision")
                continue
            else:
                node_already_in_tree = state_new_value in set(self.nodes_list_)

                if node_already_in_tree == True:
                    # Search new node
                    # print("node in tree")
                    continue
                else:
                    # print("valid node found")
                    # Valid node found
                    break

        state_new = RealVectorState(state_new_value)

        # Get nearest neighbors to state_new
        nearest_neighbors = self.get_nearest_neighbors(state_new.get_value())

        ## Add state_new to graph nodes
        self.nodes_list_.append(state_new.get_value())

        # Point with minimum cost between state_new and state_nearest
        state_min_value, cost_min = self.get_min_cost_node(state_new.get_value(), state_nearest.get_value(), nearest_neighbors)
        self.state_min_ = RealVectorState(state_min_value)

        ## Increment node count
        self.node_count_ += 1

        # state_min will be the parent node of state_new
        self.node_to_parent_[state_new.get_value()] = state_min_value
        self.node_to_cost_[state_new.get_value()] = self.node_to_cost_[state_min_value] + self.nodes_distance(state_new.get_value(), state_min_value)

        # Store the new node in the tree node map to maintain the parent pointer tree,
        # where each node has a pointer to its parent node.
        tree_parent = self.node_to_tree_node_[state_min_value]
        self.tree_nodes_.append(TreeNode(state_new, self.node_to_cost_[state_new.get_value()], tree_parent))
        self.node_to_tree_node_[state_new.get_value()] = self.tree_nodes_[-1]

        # Update tree node children for state_min -> state_new
        new_node = self.node_to_tree_node_[state_new.get_value()]
        tree_parent.add_child(new_node)

        # Rewire tree after adding new node
        self.rewire_tree(self.tree_nodes_[-1], nearest_neighbors)

        path_found = self.path_to_goal_found(state_new, self.state_goal_, self.goal_radius_)

        lower_cost_path_found = \
            self.node_to_cost_[state_new.get_value()] < self.last_cost_to_goal_

        if path_found == True and lower_cost_path_found == True:
            print("Goal node radius reached!")
            print(f"Cost: {self.node_to_cost_[state_new.get_value()]}")

            self.cost_to_goal_ = self.node_to_cost_[state_new.get_value()]

            self.last_path_found_ = self.path(state_new)
            self.last_goal_node_ = state_new
            self.last_cost_to_goal_ = self.node_to_cost_[state_new.get_value()]

        return path_found, state_nearest, state_new
    
    def run(self):
        """ Run the planner on the loaded map with no visualization until the max_number_nodes is reached.
        """
        while True:
            path_found, state_nearest, state_new = self.plan_found()
                        
            if self.max_number_nodes() == True:
                print(f"Maximum number of {self.max_num_nodes_} reached")
                break
    
    def run_test(self) -> tuple[list[tuple[int, int]], float, bool]:
        """ Run the planner on the loaded map with no visualization until a path to goal is found or until the max_number_nodes is reached.

        Returns:
            list: the path from state_init to state_goal.
            float: the cost of the path from state_init to state_goal.
            bool: true if a path to goal was found, false otherwise.
        """
        while True:
            path_found, state_nearest, state_new = self.plan_found()
                        
            if self.max_number_nodes() == True:
                print(f"Maximum number of {self.max_num_nodes_} reached")
                break
            
            if path_found == True:
                print("Path to goal found!")
                path, path_cost = self.path(state_new)
                # Log number of nodes in tree and path cost
                print(f"Number of nodes in tree: {self.node_count_}")
                print(f"Path cost: {path_cost}")
                break
        
        return path, path_cost, path_found
                
    def run_step(self):
        """Run only one step of the planner.

        Returns:
            bool: true, if a path to goal was found, false otherwise.
            State: the nearest node in tree to which the new node will
            be attached.
            State: the new node to be added to tree.
        """
        path_found, state_nearest, state_new = self.plan_found()

        return path_found, state_nearest, state_new

    def plan(self) -> tuple[list[tuple[int, int]], float]:
        """ Run the planner until the maximum number of nodes or the maximum planning time
        is reached, continuing to run_step() past the first path found so that rewiring
        keeps lowering the cost of the path to goal, instead of stopping as soon as a path
        exists.

        Returns:
            list: the path from state_init to state_goal, or an empty list if no path was found.
            float: the cost of the path, or float('inf') if no path was found.
        """
        self.start_planning_timer()

        while True:
            self.run_step()

            if self.max_number_nodes() == True:
                print(f"Maximum number of {self.max_num_nodes_} reached")
                break

            if self.max_planning_time_reached() == True:
                print(f"Maximum planning time of {self.max_planning_time_} seconds reached")
                break

        if self.last_goal_node_ is None:
            return [], float("inf")

        # last_goal_node_ is kept up to date by plan_found() every time a lower-cost path
        # is found, so this reflects the best path found up to the point the budget ran out.
        path, path_cost = self.path(self.last_goal_node_)
        print(f"Number of nodes in tree: {self.node_count_}")
        print(f"Path cost: {path_cost}")

        return path, path_cost
    
    def get_nearest_neighbors(self, node):
        """Get the nearest neighbors to the node in the tree.

        Args:
            node (tuple): the node from which we determine the nearest neighbors.

        Returns:
            set: the set of nearest neighbors to the node.
        """
        dim_configuration_space = len(node)
        eta = self.nearest_neighbor_eta_
        gamma_rrtstar = self.gamma_rrt_
        nearest_neighbors_radius = min(\
            (gamma_rrtstar * (np.log(self.node_count_) / self.node_count_ ) ** (1 / dim_configuration_space)), eta)
        # nearest_neighbors_radius = self.nearest_neighbor_radius_
        
        nearest_neighbors_estimator = NearestNeighbors(radius=nearest_neighbors_radius,
                                             algorithm='kd_tree')
        # Set training data 
        nearest_neighbors_estimator.fit(np.array(self.nodes_list_))
        
        neighbors_distance, neighbors_indexes = nearest_neighbors_estimator.radius_neighbors([np.array(list(node))])
        
        neighbors_indexes_set = np.ndenumerate(neighbors_indexes[0])
        nearest_neighbors_set = self.get_neighbors_from_nodes_list(self.nodes_list_, neighbors_indexes_set)
        
        # TODO: check if necessary this copy
        nearest_neighbors_set = copy.deepcopy(nearest_neighbors_set)
        
        return nearest_neighbors_set
        
    def get_neighbors_from_nodes_list(self, nodes_list, indexes):
        """Get the neighbors set from the node_list in tree given the indexes of the the neighbors.

        Args:
            nodes_list (_type_): the node list of the whole tree.
            indexes (_type_): the indexes of the neighbors.

        Returns:
            set: the set of neighbors.
        """
        return [nodes_list[node_num] for (x, node_num) in indexes]
    
    def cost_to_new_node(self, node1, node2):
        """Return the cost to node2 when connected to node1 in the tree.

        Args:
            node1 (tuple): the node in the tree.
            node2 (tuple): the node to be connected to node1.

        Returns:
            double: the cost to node2.
        """
        return self.node_to_cost_[node1] + self.nodes_distance(node1, node2)
    
    def get_min_cost_node(self, new_node, nearest_node, nearest_neighbors):
        """ Return node from nearest neighbors with min cost to new node.

        Args:
            new_node (tuple): the new node to be added to the tree.
            nearest_node (tuple): the nearest node to new_node in tree.
            nearest_neighbors (set of tuples): the set of nearest neighbors to new_node.

        Returns:
            tuple: node from tree with minimum cost to new_node.
            double: the minimum cost from min_cost_node to new_node.
        """
        min_cost_node = nearest_node        
        cost_min = self.cost_to_new_node(nearest_node, new_node)
        
        # Get node with minum cost to new node
        for node in nearest_neighbors:
            if self.cost_to_new_node(node, new_node) < cost_min:
                min_cost_node = node
                cost_min = self.cost_to_new_node(node, new_node)
        
        return min_cost_node, cost_min
    
    def rewire_tree(self, new_node : TreeNode, nearest_neighbors_set):
        """ Rewire tree connecting neighbors to state_new if cost is lower than current cost.

        Args:
            new_node (TreeNode): the new node added to the tree.
            nearest_neighbors_set (set of tuples): the set of nearest neighbors to new node.
        """
        # Get the newly added node coordinates for easier access
        new_node_coords = new_node.get_node_coordinates()

        # Check if each neighbor can get a lower cost by connecting to state_new
        for near_node in nearest_neighbors_set:
            node_to_rewire = self.node_to_tree_node_[near_node]
            node_to_rewire_coords = node_to_rewire.get_node_coordinates()
            node_to_rewire_parent = None
            
            # Check if near_node can get lower cost by connecting to new_node
            if self.nodes_closer(new_node_coords, near_node) == True:
                node_to_rewire_parent = node_to_rewire.get_parent()
                
                # Remove near_node from its current parent children list
                node_to_rewire_parent.remove_child(node_to_rewire)
                
                # Update parent and cost of near_node to connect to new_node
                self.node_to_parent_[near_node] = new_node_coords
                self.node_to_cost_[near_node] = self.cost_to_new_node(new_node_coords, node_to_rewire_coords)
                
                # Update tree node parent and cost for near_node to connect to new_node
                new_node.add_child(node_to_rewire)
                node_to_rewire.set_parent(new_node)
                
                # Update cost of new_node to reflect the new connection
                new_node_cost = self.node_to_cost_[new_node_coords]
                new_node.set_cost(new_node_cost)
                
                # Update cost of child nodes of near_node to reflect the new connection
                self.update_child_costs(node_to_rewire)
                
    def update_child_costs(self, node_to_rewire):
        """ Update the cost of the child nodes of the rewired node to reflect the new connection.

        Args:
            node_to_rewire (TreeNode): the node that was rewired to connect to new_node.
        """
        # Get the child nodes of the rewired node
        child_nodes = node_to_rewire.get_children()
        
        # Recursively update the cost of the child nodes
        for child_node in child_nodes:
            child_node_coords = child_node.get_node_coordinates()
            node_to_rewire_coords = node_to_rewire.get_node_coordinates()
            
            # Update cost of child node to reflect the new connection
            self.node_to_cost_[child_node_coords] = self.cost_to_new_node(node_to_rewire_coords, child_node_coords)
            
            # Recursively update the cost of the child nodes of the child node
            self.update_child_costs(child_node)
                
    def nodes_closer(self, new_node, tree_node):
        """ Return if new_node appended to tree_node has lower cost than the cost from tree_node itself.

        Args:
            new_node (tuple): the new node to be added in tree.
            tree_node (tuple): the node already in the tree.

        Returns:
            bool: True if new_node appended to tree_node yields lower cost than tree_node itself.
        """
        nodes_are_closer = self.node_to_cost_[new_node] + self.nodes_distance(new_node, tree_node) < self.node_to_cost_[tree_node]
        return nodes_are_closer
        