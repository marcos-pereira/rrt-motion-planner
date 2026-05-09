class TreeBuilder:
    def __init__(self, root_node: tuple[int, int]):
        self.adjacency_list_ = dict[tuple[int, int], list[tuple[int, int]]]()
        self.adjacency_list_[root_node] = list()
        
    def add_node(self, parent_node: tuple[int, int], new_node: tuple[int, int]):
        if parent_node in self.adjacency_list_:
            self.adjacency_list_[parent_node].append(new_node)
            if new_node not in self.adjacency_list_:
                self.adjacency_list_[new_node] = list()
        else:
            raise ValueError(f"Parent node {parent_node} not found in the tree. Connectivity between nodes and edges cannot be maintained.")