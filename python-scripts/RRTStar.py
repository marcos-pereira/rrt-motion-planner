from RRTPlanner import RRTPlanner
from sklearn.neighbors import NearestNeighbors
import numpy as np
import copy

class RRTStar(RRTPlanner):
    def __init__(self,
                 x_init, 
                 x_goal, 
                 goal_radius, 
                 steer_delta, 
                 nearest_neighbor_eta,
                 gamma_rrt,
                 nearest_neighbor_radius,
                 scene_map, 
                 max_num_nodes):
        """ Return RRTStar planner.

        Args:
            x_init (_type_): The initial configuration node.
            x_goal (_type_): The goal configuration node.
            goal_radius (_type_): Radius to be considered within the goal.
            steer_delta (_type_): Value used to steer toward the sampled configurations.
            gamma_rrt (_type_): Gain used to determine radius of ball for nearest neighbors
            scene_map (_type_): Map of the scene or configuration space.
            max_num_nodes (_type_): Maximum number of nodes in the tree.
        """
        super().__init__(x_init, 
                         x_goal, 
                         goal_radius, 
                         steer_delta, 
                         scene_map, 
                         max_num_nodes)
        
        self.gamma_rrt_ = gamma_rrt
        self.nearest_neighbor_eta_ = nearest_neighbor_eta
        self.nearest_neighbor_radius_ = nearest_neighbor_radius
        
        # Initial cost to goal
        self.cost_to_goal_ = np.inf
        
        ## Store last path found
        self.last_path_found_ = list()

        ## Store last point in goal
        self.last_goal_found_ = tuple()
        
        # Remove this edges
        self.remove_this_edges_ = list()
        
        # Rewire this edges
        self.rewired_edges_ = list()
        
        # Node with minimum cost in tree to connect new node to
        self.x_min_ = tuple()
        
    def plan_found(self):
        """ Returns if a plan could be found, the nearest node to the newest node, and the new node.

        Returns:
            bool: True if a plan is found, false otherwise.
            tuple: the nearest node to the new node added.
            tuple: the new node found.
        """
                
        while True:
            x_rand = self.configuration_in_free_space()
            
            # Get nearest node to x_rand
            x_nearest = self.nearest_node(x_rand, self.rrt_graph_)
            
            # Steer from nearest node in tree towards the x_rand 
            # to obtain a new node for the tree
            # x_new = self.steer(x_nearest, x_rand, self.steer_delta_)
            x_new = self.linear_interpolation(x_nearest, x_rand, self.steer_delta_)
            
            if not x_new:
                # Cannot connect
                continue
            else:
                node_already_tree = x_new in set(self.nodes_list_)
                
                if node_already_tree:
                    # Search for new node
                    continue
                else:
                    # Valid node found
                    break
            
        ## x_nearest will be the parent node of x_new
        self.node_to_parent_[x_new] = x_nearest

        self.node_to_cost_[x_new] = self.cost_to_node(x_new)        
        
        # Get nearest neighbors to x_new 
        nearest_neighbors = self.get_nearest_neighbors(x_new)
        
        ## Add x_new to graph
        self.insert_node_to_tree(x_new, 0)
        self.nodes_list_.append(x_new)
        
        ## Increment node count
        self.node_count_ += 1
        
        # Point with minimum cost between x_new and x_nearest
        x_min, cost_min = self.get_min_cost_node(x_new, x_nearest, nearest_neighbors)
        self.x_min_ = x_min
        
        # Add edge between x_min and x_new
        self.add_edge(x_min, x_new)
        self.node_to_parent_[x_new] = x_min
        self.node_to_cost_[x_new] = cost_min
        
        # Rewire tree after adding new node
        self.rewire_tree(x_new, nearest_neighbors)
        
        path_found = self.path_to_goal_found(x_new, self.x_goal_, self.goal_radius_)
        
        lower_cost_path_found = self.node_to_cost_[x_new] < self.cost_to_goal_
        
        if path_found and lower_cost_path_found:
            self.cost_to_goal_ = self.node_to_cost_[x_new]
            print(f"Cost: {self.cost_to_goal_}")
            
            ## At least one path was found
            self.one_path_found_ = True

            ## Goal was found, get path to goal
            self.last_path_found_ = self.path()
            self.last_goal_found_ = x_new
            
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
                print(f"Cost to goal: {self.cost_to_goal_}")
    
    def run_step(self):
        path_found, x_nearest, x_new = self.plan_found()
        
        return path_found, x_nearest, x_new
    
    def get_nearest_neighbors(self, node):
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
        # nearest_neighbors_set = copy.deepcopy(nearest_neighbors_set)
        
        return nearest_neighbors_set
        
    def get_neighbors_from_nodes_list(self, nodes_list, indexes):
        return [nodes_list[node_num] for (x, node_num) in indexes]
    
    def cost_between_nodes(self, node1, node2):
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
        cost_min = self.cost_between_nodes(nearest_node, new_node)
        
        # Get node with minum cost to new node
        for node in nearest_neighbors:
            if self.cost_between_nodes(node, new_node) < cost_min:
                min_cost_node = node
                cost_min = self.cost_between_nodes(node, new_node)
        
        return min_cost_node, cost_min
    
    def rewire_tree(self, x_new, nearest_neighbors_set):
        """ Rewire tree connecting nodes in tree to x_new if cost
        is lower than current cost. Remove the old edges and add new ones.

        Args:
            x_new (tuple): the new node to in tree.
            nearest_neighbors_set (set of tuples): the set of nearest neighbors to new node.
        """
        self.remove_this_edges_ = list()
        for node in nearest_neighbors_set:
            x_parent = None
            if self.nodes_closer(x_new, node):
                # x_parent = copy.deepcopy(self.node_to_parent_[node])
                x_parent = self.node_to_parent_[node]
                                
            ## Delete edge between node and its parent
            ## and rewire it with x_new since the cost is smaller
            if x_parent is not None:
                # Remove edge
                self.rrt_graph_[1].remove((x_parent, node))
                self.remove_this_edges_.append((x_parent, node))
                
                self.add_edge(x_new, node)
                self.rewired_edges_.append((x_new, node))
                self.node_to_parent_[node] = x_new
                self.node_to_cost_[node] = self.cost_to_node(node)
                self.node_to_cost_[x_new] = self.cost_to_node(x_new)

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
        