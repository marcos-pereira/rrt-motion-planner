#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.
  
#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors: 
# marcos-pereira (https://github.com/marcos-pereira)

from RRTPlanner import RRTPlanner
from RealVectorState import RealVectorState
from State import State
from Steer import Steer
from TreeNode import TreeNode

class RRT(RRTPlanner):
    def __init__(self,
                 state_init: State,
                 state_goal: State,
                 goal_radius,
                 steer_delta,
                 steer: Steer,
                 scene_map,
                 max_num_nodes,
                 max_planning_time=None):
        """Return an RRT planner object that plans by running the method run() or that
        plans only one iteration by running the method run_step().

        Args:
            state_init (State): the initial configuration.
            state_goal (State): the goal configuration.
            goal_radius (double): the radius of a ball around the goal configuration.
            steer_delta (int): the steer step towards goal when going from a node in the tree
            towards the new node being added. This parameter is map dependant and will vary
            for each map.
            steer (Steer): the steering strategy used to move from a node in the tree
            towards the new node being added.
            scene_map (numpy matrix): the binary matrix where 0 indicate free space and 1
            indicate an obstacle.
            max_num_nodes (_type_): maximum number of nodes to be sampled. The planner stops
            when this number is reached.
            max_planning_time (float): the maximum time in seconds that plan() may run, or
            None to only bound the search by max_num_nodes.
        """
        super().__init__(state_init,
                         state_goal,
                         goal_radius,
                         steer_delta,
                         steer,
                         scene_map,
                         max_num_nodes,
                         max_planning_time)
        
    def plan_found(self) -> tuple[bool, State, State]:
        """ Returns if a plan could be found, the nearest node to the newest node, and the new node.

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
            if self.collision(state_new_value):
                # Node in collision
                # print("collision")
                continue
            else:
                node_already_in_tree = state_new_value in set(self.nodes_list_)

                if node_already_in_tree:
                    # Search new node
                    # print("node in tree")
                    continue
                else:
                    # print("valid node found")
                    # Valid node found
                    break

        state_new = RealVectorState(state_new_value)

        ## state_nearest will be the parent node of state_new
        self.node_to_parent_[state_new.get_value()] = state_nearest.get_value()

        self.node_to_cost_[state_new.get_value()] = self.cost_to_node(state_new.get_value())

        ## Add state_new to graph nodes
        self.nodes_list_.append(state_new.get_value())

        ## Increment node count
        self.node_count_ += 1

        self.add_edge(state_nearest.get_value(), state_new.get_value())

        # Add node to tree builder to keep track of the neighbors of each node in the graph
        self.tree_builder_.add_node(state_nearest.get_value(), state_new.get_value())

        # Store the new node in the tree node map to maintain the parent pointer tree,
        # where each node has a pointer to its parent node.
        tree_parent = self.node_to_tree_node_[state_nearest.get_value()]
        self.tree_nodes_.append(TreeNode(state_new, self.node_to_cost_[state_new.get_value()], tree_parent))
        self.node_to_tree_node_[state_new.get_value()] = self.tree_nodes_[-1]

        path_found = self.path_to_goal_found(state_new, self.state_goal_, self.goal_radius_)

        return path_found, state_nearest, state_new
    
    def run(self) -> tuple[list[tuple[int, int]], float]:
        """ 
        Run the RRT planner until a path to goal is found or until the maximum number of nodes is reached.
        Returns:
            list: the path from state_init to state_goal.
            float: the cost of the path from state_init to state_goal.
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
            
        return path, path_cost
    
    def run_test(self) -> tuple[list[tuple[int, int]], float, bool]:
        """ Run the RRT planner until a path to goal is found or until the maximum number of nodes is reached. 
        This method is used for testing purposes, as it also returns if a path to goal was found or not.

        Returns:
            list: the path from state_init to state_goal.
            float: the cost of the path from state_init to state_goal.
            bool: true, if a path to goal was found, false otherwise.
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
        """ Run the RRT planner until a path to goal is found or until the maximum number of
        nodes or the maximum planning time is reached. Unlike run(), this always returns a
        well-defined path and cost, using an empty path and infinite cost to signal a timeout.

        Returns:
            list: the path from state_init to state_goal, or an empty list if no path was found.
            float: the cost of the path, or float('inf') if no path was found.
        """
        self.start_planning_timer()

        while True:
            path_found, state_nearest, state_new = self.run_step()

            if path_found == True:
                print("Path to goal found!")
                path, path_cost = self.path(state_new)
                print(f"Number of nodes in tree: {self.node_count_}")
                print(f"Path cost: {path_cost}")
                return path, path_cost

            if self.max_number_nodes() == True:
                print(f"Maximum number of {self.max_num_nodes_} reached")
                return [], float("inf")

            if self.max_planning_time_reached() == True:
                print(f"Maximum planning time of {self.max_planning_time_} seconds reached")
                return [], float("inf")