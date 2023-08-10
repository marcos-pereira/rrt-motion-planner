#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.
  
#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors: 
# marcos-pereira (https://github.com/marcos-pereira)

from RRTPlanner import RRTPlanner

class RRT(RRTPlanner):
    def __init__(self, 
                 x_init, 
                 x_goal, 
                 goal_radius, 
                 steer_delta, 
                 scene_map, 
                 max_num_nodes):
        super().__init__(x_init, 
                         x_goal, 
                         goal_radius, 
                         steer_delta, 
                         scene_map, 
                         max_num_nodes)
        
    def plan_found(self):
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
        
        path_found = self.path_to_goal_found(x_new, self.x_goal_, self.goal_radius_)
        
        return path_found, x_nearest, x_new
    
    def run(self):
        """ Run the planner on the loaded map with no visualization.
        """
        
        while True:            
            path_found = self.plan_found()
            
            if self.max_number_nodes():
                print(f"Maximum number of {self.max_num_nodes_} reached")
                break
            
            if path_found:
                print("Path to goal found!")
                break
    
    def run_step(self):
        path_found, x_nearest, x_new = self.plan_found()
        
        return path_found, x_nearest, x_new