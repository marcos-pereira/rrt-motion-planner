from abc import ABC, abstractmethod
import numpy as np
import random
from rtree import index

class RRTPlanner(ABC):
    def __init__(self,
                 x_init,
                 x_goal,
                 goal_radius,
                 steer_delta,
                 scene_map,
                 max_num_nodes):
        self.x_init_ = x_init
        self.x_goal_ = x_goal
        self.goal_radius_ = goal_radius
        self.steer_delta_ = steer_delta
        self.max_num_nodes_ = max_num_nodes
        self.scene_map_ = scene_map

        self.map_height_, self.map_width_ = self.scene_map_.shape
        
        # interleaved True: requires coordinates as [xmin ymin, xmax ymax]
        # See: https://rtree.readthedocs.io/en/latest/class.html#rtree.index.Property
        # index.Property: inherits some instation properties:
        # See: https://rtree.readthedocs.io/en/latest/class.html#rtree.index.Property
        self.nodes_ = index.Index(interleaved=True, properties=index.Property())

        ## Used to get nearest neighbors using KDtree
        self.nodes_list_ = list()
        self.nodes_list_.append(self.x_init_)
        
        self.node_count_ = 1

        self.edges_ = set()

        ## Initialize costs map
        self.node_to_cost_ = dict()
        self.node_to_cost_[self.x_init_] = 0

        ## Initialize parent map
        self.node_to_parent_ = dict()
        self.node_to_parent_[self.x_init_] = self.x_init_

        ## Initalize graph
        self.rrt_graph_ = (self.nodes_, self.edges_)
        
        x_init_id = 0
        self.insert_node_to_tree(x_init, x_init_id)
        
        ## Used to detect collisions with obstacles
        self.ones_in_drawing_ = np.where(self.scene_map_ == 1)
        self.obstacles_coordinates_ = set(zip(self.ones_in_drawing_[1], self.ones_in_drawing_[0]))
        
        ## Initialize path to goal empty
        self.path_ = list()
        
        ## New point to add to tree
        self.x_new_ = tuple()
        
        ## Store if path was found
        self.path_found_ = False
        
    @abstractmethod
    def plan_found(self):
        pass
    
    @abstractmethod
    def run(self):
        pass
    
    @abstractmethod
    def run_step(self):
        pass
    
    def run_test(self):
        """ Run the planner on the loaded map with no visualization.
        """
        path_found = False
        while True:
            self.run_planner_step()
            
            path_found = self.path_to_goal_found()
            
            if self.max_number_nodes() == True:
                break
            
            if path_found == True:
                print("Path to goal found!")
                break
    
    def path_to_goal_found(self, x_new, x_goal, goal_radius):
        """ Returns if the path to goal was found.

        Args:
            x_new (_type_): is the new node added to the tree.
            x_goal (_type_): is the goal node in the map.
            goal_radius (_type_): is the radius that considers the
            goal node was reached.

        Returns:
            bool: True if the path was found, false otherwise.
        """
        path_found = False
        
        ## Check if goal radius was reached
        if self.nodes_distance(x_new, x_goal) < goal_radius:
            print("Goal node radius reached!")     
                                        
            path_found = True
        
        return path_found
    
    def max_number_nodes(self):
        """ Check if maximum number of nodes was reached.

        Returns:
            bool: True if maximum number of nodes was reached.
        """
        
        max_number_nodes_reached = False
        
        if self.node_count_ >= self.max_num_nodes_:
            print("Maximal number of nodes in tree reached!")
            print("Input anything and press enter to quit.")
            max_number_nodes_reached = True
            
            return max_number_nodes_reached
        else:
            return max_number_nodes_reached
        
    def sample_space(self, x_max, y_max):
        """ Sample the configuration space with limits x_max and y_max.

        Args:
            x_max (int): the maximum x coordinate.
            y_max (int): the maximum y coordinate.

        Returns:
            tuple: the sampled tuple configuration.
        """
        x = random.randint(0, x_max)
        y = random.randint(0, y_max)

        x_rand = (x, y)

        return x_rand
    
    def nodes_distance(self, node1, node2):
        """ Returns the distance between node1 and node2.

        Args:
            node1 (tuple): the first node.
            node2 (tuple): the second node.

        Returns:
            double: the norm of the vector between node2 and node1.
        """
        p1 = np.array([node1[0], node1[1]])
        p2 = np.array([node2[0], node2[1]])
        distance = np.linalg.norm(p1 - p2)

        return distance
    
    def steer(self, node1, node2, delta):
        """ Returns a node between node1 and node2. If they are close by delta, then 
        return node2.

        Args:
            node1 (tuple): the initial node.
            node2 (tuple): the goal node towards which we steer.
            delta (double): the minimum distance to consider already near enough to node2.

        Returns:
            tuple: the new node between node1 and node2. 
        """
        node1 = np.array([node1[0], node1[1]])
        node2 = np.array([node2[0], node2[1]])
        if self.nodes_distance(node1, node2) < delta:
            node = node2
        else:
            diffnodes = node2 - node1
            diffnodes = diffnodes/self.nodes_distance(node1, node2)
            node = node1 + delta*diffnodes

        # Convert to int, otherwise the maps will not work with double precision
        # TODO: use some better mapping like a hash function to avoid this problem
        node = tuple(int(element) for element in node)

        return  node

    def linear_interpolation(self, node1, node2, delta):
        
        node1 = np.array([node1[0], node1[1]])
        node2 = np.array([node2[0], node2[1]])

        for interpolation_factor in np.arange(0, 1, delta):
            node = node1*interpolation_factor + (1-interpolation_factor)*node2
            node = tuple(element for element in node)    
            if self.collision(node) == True:
                return False        
        
        # Convert to int, otherwise the maps will not work with double precision
        # TODO: use some better mapping like a hash function to avoid this problem
        node = tuple(int(element) for element in node)
        
        return node
    
    def path(self, node):
        """ Get path from node to initial node using the map node_to_parent. 
        Return also the path cost.

        Returns:
            list: list of tuples that associate the node and its parent node.
            doube: path cost.
        """

        ## Current node starts as the last node of the trajectory
        current_node = node
        
        path = list()

        while True:
            ## Add tuple of current node to the path
            path.append(current_node)

            ## Current node become its parent, we are searching for a path
            ## backwards in the tree
            current_node = self.node_to_parent_[current_node]

            ## If x_init is reached
            if current_node[0] == self.x_init_[0] and \
                current_node[1] == self.x_init_[1]:
                break
        
        path_cost = self.cost_to_node(node)

        return path, path_cost
    
    def nearest_node(self, current_node, rrt_graph):
        """ Get nearest node to current node in the rrt_graph.

        Args:
            current_node (tuple): the current node.
            rrt_graph (rtree index): the rrt graph.

        Returns:
            tuple: the nearest node to current_node.
        """
        
        # Number of nearest neighbors to query
        num_nearest_neighbors = 1
        
        # The raw object is the node itself as it was inserted in the rtree
        return_raw_object_from_rtree = "raw"
        
        # Get the nearest node (the first element of the rrt_graph is the node tree)
        nearest_node_pair = rrt_graph[0].nearest(current_node, num_results=num_nearest_neighbors, objects=return_raw_object_from_rtree)
        nearest_node_pair_as_list = list(nearest_node_pair)
        
        return nearest_node_pair_as_list[0]

    def configuration_in_free_space(self):
        """Get a configuration in the free configuration space.

        Returns:
            tuple: the configuration in the configuration free space.
        """
        ## Sample configuration in space
        x_rand = self.sample_space(self.map_width_, self.map_height_)
        
        # Sample until no collision occurs
        while x_rand in self.obstacles_coordinates_:
            x_rand = self.sample_space(self.map_width_, self.map_height_)
            
        return x_rand
    
    def cost_to_node(self, node): 
        """ Return the cost from initial node in the tree to the node.

        Args:
            node (tuple): the node to be added to tree.

        Returns:
            double: the cost from initial node in the tree to the node.
        """
        parent_node = self.node_to_parent_[node]
        cost_to_parent = self.node_to_cost_[parent_node]
        cost_parent_to_node = self.nodes_distance(parent_node, node)
        cost = cost_to_parent + cost_parent_to_node
        
        return cost

    def collision(self, node):
        """ Check if node is in collision. First the node is converted to integers
        because the configuration space is discretized into integers.

        Args:
            node (tuple): The node to check if collides with obstacles.

        Returns:
            bool: True if in collision, false otherwise.
        """
        node_integers = tuple(int(element) for element in node)
        if node_integers in self.obstacles_coordinates_:
            return True
        else:
            return False
        
    def insert_node_to_tree(self, node, node_id=0):
        """ Insert node to rrt_tree node tree.

        Args:
            node (tuple): the node to be inserted to the rrt_tree node tree.
            node_id (int): the id of the inserted node.
        """
        
        # The rtree requires the doubled coordinates of the object to be stored
        # See the insert documentation: https://rtree.readthedocs.io/en/latest/class.html#rtree.index.Property
        node_coordinates_doubled = node + node
        
        # The first element of the rrt_graph is the node tree
        self.rrt_graph_[0].insert(node_id, node_coordinates_doubled, node)
        
    def add_edge(self, node1, node2):
        """ Add edge from node 1 to node 2 in the rrt_graph.

        Args:
            node1 (tuple): The first node.
            node2 (tuple): The second node.
        """
        
        # The second element of rrt_graph is the edge list
        self.rrt_graph_[1].add((node1, node2))
        
    def get_graph(self):
        """Return the RRT graph.
        """
        
        return self.rrt_graph_