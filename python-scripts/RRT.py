#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.
  
#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors: 
# marcos-pereira (https://github.com/marcos-pereira)

from RRTPlanner import RRTPlanner
from TreeNode import TreeNode

class RRT(RRTPlanner):
    def __init__(self, 
                 x_init, 
                 x_goal, 
                 goal_radius, 
                 steer_delta, 
                 scene_map, 
                 max_num_nodes):
        """Return an RRT planner object that plans by running the method run() or that 
        plans only one iteration by running the method run_step().

        Args:
            x_init (tuple): the initial configuration.
            x_goal (tuple): the goal configuration.
            goal_radius (double): the radius of a ball around the goal configuration.
            steer_delta (int): the steer step towards goal when going from a node in the tree
            towards the new node being added. This parameter is map dependant and will vary
            for each map.
            scene_map (numpy matrix): the binary matrix where 0 indicate free space and 1
            indicate an obstacle.
            max_num_nodes (_type_): maximum number of nodes to be sampled. The planner stops
            when this number is reached.
        """
        super().__init__(x_init, 
                         x_goal, 
                         goal_radius, 
                         steer_delta, 
                         scene_map, 
                         max_num_nodes)
        
    def plan_found(self) -> tuple[bool, tuple[int, int], tuple[int, int]]:
        """ Returns if a plan could be found, the nearest node to the newest node, and the new node.

        Returns:
            bool: True if a plan is found, false otherwise.
            tuple: the nearest node to the new node added.
            tuple: the new node found.
        """
        
        while True:
            x_rand = self.configuration_in_free_space()
            
            ## Get nearest node to x_rand
            x_nearest = self.nearest_node(x_rand, self.rrt_graph_)

            ## Steer from nearest node in tree (i.e. parent_node) towards the
            ## x_rand to obtain a new node for the tree
            x_new = self.steer(x_nearest, x_rand, self.steer_delta_)
            
            ## Check if node is in collision
            if self.collision(x_new):
                # Node in collision
                # print("collision")
                continue
            else:
                node_already_in_tree = x_new in set(self.nodes_list_)
                
                if node_already_in_tree:
                    # Search new node
                    # print("node in tree")
                    continue
                else:
                    # print("valid node found")
                    # Valid node found                    
                    break

        ## x_nearest will be the parent node of x_new
        self.node_to_parent_[x_new] = x_nearest
        
        self.node_to_cost_[x_new] = self.cost_to_node(x_new)            

        ## Add x_new to graph
        self.insert_node_to_tree(x_new, 0)
        self.nodes_list_.append(x_new)
        
        ## Increment node count
        self.node_count_ += 1

        self.add_edge(x_nearest, x_new)
                    
        # Add node to tree builder to keep track of the neighbors of each node in the graph
        self.tree_builder_.add_node(x_nearest, x_new)
        
        # Store the new node in the tree node map to maintain the parent pointer tree, 
        # where each node has a pointer to its parent node.
        tree_parent = self.node_to_tree_node_[x_nearest]
        self.tree_nodes_.append(TreeNode(x_new, self.node_to_cost_[x_new], tree_parent))
        self.node_to_tree_node_[x_new] = self.tree_nodes_[-1]
                
        path_found = self.path_to_goal_found(x_new, self.x_goal_, self.goal_radius_)
        
        return path_found, x_nearest, x_new
    
    def run(self) -> tuple[list[tuple[int, int]], float]:
        """ 
        Run the RRT planner until a path to goal is found or until the maximum number of nodes is reached.
        Returns:
            list: the path from x_init to x_goal.
            float: the cost of the path from x_init to x_goal.
        """
        
        while True:            
            path_found, x_nearest, x_new = self.plan_found()
            
            if self.max_number_nodes() == True:
                print(f"Maximum number of {self.max_num_nodes_} reached")
                break
            
            if path_found == True:
                print("Path to goal found!")                
                path, path_cost = self.path(x_new)
                # Log number of nodes in tree and path cost
                print(f"Number of nodes in tree: {self.node_count_}")
                print(f"Path cost: {path_cost}")
                break
            
        return path, path_cost
    
    def run_step(self):
        """Run only one step of the planner.

        Returns:
            bool: true, if a path to goal was found, false otherwise.
            tuple: the nearest node in tree to which the new node will
            be attached.
            tuple: the new node to be added to tree.
        """
        path_found, x_nearest, x_new = self.plan_found()
        
        return path_found, x_nearest, x_new