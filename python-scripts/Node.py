class Node:
    def __init__(self,
                 point,
                 parent,
                 index,
                 children,
                 cost):
        self.point_ = point
        self.parent_ = parent
        self.index_ = index
        self.children_ = children
        self.cost_ = cost

    def point(self):
        return self.point_

    def parent(self):
        return self.parent_

    def index(self):
        return self.index_

    def children(self):
        return self.children_

    def cost(self):
        return self.cost_
