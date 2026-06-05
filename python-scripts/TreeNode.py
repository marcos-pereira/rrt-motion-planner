class TreeNode:
    """ Class to represent the tree node in the data structure used in RRT and RRT* planners. 
    This node will be used to store the parent-pointer tree, where each node has 
    a pointer to its parent node.
    """
    def __init__(self, root_node_coordinates: tuple[int, int], cost: float, parent_node):
        # The node is represented as a tuple of (x, y) coordinates.
        self.node_coordinates_ = root_node_coordinates
        
        # Cost to reach this node from the root node. This will be used in RRT* planner to
        # compute the cost of the path from the root node to this node. 
        self.cost_ = cost
        
        # Pointer to the parent node in the tree. 
        # This will be used to reconstruct the path from the 
        # goal node to the root node.
        self.parent_ = parent_node
        
    def get_parent(self):
        """ Returns the parent node of this tree node. """
        return self.parent_
    
    def get_node_coordinates(self) -> tuple[int, int]:
        """ Returns the node coordinates of this tree node. """
        return self.node_coordinates_
    
    def get_cost(self) -> float:
        """ Returns the cost of this tree node. """
        return self.cost_
        
        
        
    