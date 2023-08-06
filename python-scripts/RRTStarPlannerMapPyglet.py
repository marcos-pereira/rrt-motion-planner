import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import pygame
pygame.init()
from pyglet import *
from pyglet.gl import *
from Node import Node
from rtree import index
from sklearn.neighbors import NearestNeighbors
import time
from time import sleep

class Line():
    def __init__(self, x1, y1, x2, y2, batch, group):
        self.line_ = shapes.Line(x1, y1, x2, y2, color=(196, 0, 183), width=2, batch=batch, group=group)

class Path():
    def __init__(self, x1, y1, x2, y2, batch, group):
        self.path_ = shapes.Line(x1, y1, x2, y2, color=(53, 155, 192), width=4, batch=batch, group=group)

class RRTStarPlanner(pyglet.window.Window):
    def __init__(self,
                 x_init,
                 x_goal,
                 goal_radius,
                 steer_delta,
                 map_name,
                 scene_map,
                 search_window,
                 width,
                 height,
                 num_nodes,
                 font_size,
                 *args,
                 **kwargs):
        super(RRTStarPlanner, self).__init__(width, height, *args,
                                         **kwargs)
        print("Initializing RRTPlanner...")
        self.x_init_ = x_init
        self.x_goal_ = x_goal
        self.goal_radius_ = goal_radius
        self.steer_delta_ = steer_delta
        self.search_window_ = search_window
        self.num_nodes_ = num_nodes
        self.font_size_ = font_size
        self.map_name_ = map_name
        self.scene_map_ = scene_map
        self.cost_to_goal_ = np.Inf

        self.nodes_ = index.Index(interleaved=True, properties=index.Property())
        self.nodes_.insert(0, self.x_init_ + self.x_init_, self.x_init_)

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

        self.map_height_, self.map_width_ = self.scene_map_.shape

        ## Used to detect collisions with obstacles
        self.ones_in_drawing_ = np.where(self.scene_map_ == 1)
        self.obstacles_coordinates_ = set(zip(self.ones_in_drawing_[1], self.ones_in_drawing_[0]))

        ## Initialize path to goal empty
        self.path_ = list()

        ## Pyglet
        self.batch_ = pyglet.graphics.Batch()
        self.background_ = pyglet.graphics.Group(order=0)
        self.foreground_ = pyglet.graphics.Group(order=1)
        self.image_layer_ = pyglet.graphics.Group(order=2)
        self.path_layer_ = pyglet.graphics.Group(order=3)

        ## Store lines
        self.lines_ = dict()

        ## Store path to draw
        self.path_line_ = set()

        ## Control window behavior
        self.alive_ = 0
        self.finish_ = 0

        ## Load map
        self.map_ = image.load(map_name)
        self.map_sprite_ = pyglet.sprite.Sprite(self.map_, 0, 0, batch=self.batch_, group=self.image_layer_)

        ## Store if path was found
        self.path_found_ = False
        self.one_path_found_ = False

        ## Store last path found
        self.last_path_found_ = list()

        ## Store last point in goal
        self.last_goal_point_ = tuple()

        ## Key press
        self.key_ = pyglet.window.key

        ## Store path cost text
        self.path_cost_text_ = pyglet.text.Label(
            'Path cost: ' + str('%.2f' % self.cost_to_goal_),
            font_name='Arial',
            font_size=self.font_size_, x=0, y=self.font_size_,
            batch=self.batch_, group=self.path_layer_)

    def on_key_press(self, symbol, modifiers):
        if symbol == self.key_.ESCAPE:
            self.alive_ = 0
            self.finish_ = 1
            self.close()

        if symbol == self.key_.S:
            self.alive_ = 1

    def run(self):
        path_found = False
        while True:
            if self.alive_ == 1:

                if not path_found:
                    path_found = self.plan_iteration()

                # -----------> This is key <----------
                # This is what replaces pyglet.app.run()
                # but is required for the GUI to not freeze
                #

            if self.finish_ == 1:
                return

            event = self.dispatch_events()

        return

    def plan_iteration(self):
        self.clear()

        x_new = tuple()

        if self.path_found_ == False:
            ## Sample point in space
            x_rand = self.sample_space(self.map_width_, self.map_height_)

            ## Check if point is inside any obstacle
            if x_rand in self.obstacles_coordinates_:
                return False

            ## Get nearest node to x_rand
            x_nearest = self.nearest(x_rand, self.x_init_, self.rrt_graph_)

            ## Steer from nearest node in tree (i.e. parent_node) towards the
            ## x_rand to obtain a new node for the tree
            x_new = self.steer(x_nearest, x_rand, self.steer_delta_)
            ## Check if x_new is already in tree
            if x_new in set(self.nodes_list_):
                return False

            ## x_nearest will be the parent node of x_new
            self.node_to_parent_[x_new] = x_nearest

            ## Cost to x_new
            self.node_to_cost_[x_new] = self.node_to_cost_[self.node_to_parent_[x_new]] + \
                                        self.nodes_distance(self.node_to_parent_[x_new], x_new)

            ## Check if point is inside any obstacle
            if (int(x_new[0]), int(x_new[1])) in self.obstacles_coordinates_:
                return False

            ## Get nodes near to x_new
            near_d = 10
            near_eta = self.steer_delta_
            gamma_rrt = 10
            near_radius = min((gamma_rrt * (
                        np.log(self.node_count_) / self.node_count_ ** (1 / near_d)),
                               near_eta))
            nearest_neighbors = NearestNeighbors(radius=near_radius,
                                                 algorithm='kd_tree')
            nearest_neighbors.fit(np.array(self.nodes_list_))
            neighbors = nearest_neighbors.radius_neighbors(
                [np.array(list(x_new))])
            neighbors_indexes = neighbors[1][0]
            x_near = [self.nodes_list_[y] for (x, y) in
                      np.ndenumerate(neighbors_indexes)]
            x_near = copy.deepcopy(x_near)
            # x_near = self.near(x_new, nodes_set, 10, near_radius)

            ## Add x_new to graph
            # rrt_graph[0].add(x_new)
            self.rrt_graph_[0].insert(0, x_new + x_new, x_new)
            self.nodes_list_.append(x_new)

            ## Increment node count
            self.node_count_ += 1

            ## Point with minimum cost between x_new and x_nearest
            x_min = x_nearest
            cost_min = self.node_to_cost_[x_nearest] + self.nodes_distance(x_nearest,
                                                                     x_new)
            for node in x_near:
                if self.node_to_cost_[node] + self.nodes_distance(node,
                                                            x_new) < cost_min:
                    # print("reduce cost")
                    x_min = node
                    cost_min = self.node_to_cost_[node] + self.nodes_distance(node,
                                                                        x_new)

            ## Add edge between x_min and x_new
            self.rrt_graph_[1].add((x_min, x_new))
            self.node_to_parent_[x_new] = x_min
            self.node_to_cost_[x_new] = cost_min

            self.lines_[(x_min, x_new)] = \
                Line(x_min[0], self.map_height_ - x_min[1],
                     x_new[0], self.map_height_ - x_new[1],
                     self.batch_, self.foreground_)

            ## Rewire tree
            for node in x_near:
                x_parent = None
                if self.node_to_cost_[x_new] + self.nodes_distance(x_new, node) < \
                        self.node_to_cost_[node]:
                    x_parent = copy.deepcopy(self.node_to_parent_[node])
                ## Delete edge between node and its parent
                ## and rewire it with x_new since the cost is smaller
                if x_parent is not None:
                    # print("Rewire")
                    self.rrt_graph_[1].remove((x_parent, node))
                    self.lines_.pop((x_parent, node))
                    self.rrt_graph_[1].add((x_new, node))
                    self.lines_[(x_new, node)] = \
                        Line(x_new[0], self.map_height_ - x_new[1],
                             node[0], self.map_height_ - node[1],
                             self.batch_, self.foreground_)
                    self.node_to_parent_[node] = x_new
                    self.node_to_cost_[node] = self.node_to_cost_[
                                             x_new] + self.nodes_distance(x_new,
                                                                          node)

        ## Draw x_init and x_goal
        draw_x_init = shapes.Circle(self.x_init_[0], self.map_height_-self.x_init_[1], radius=10, color=(255, 207, 88), batch=self.batch_, group=self.foreground_)
        draw_x_goal = shapes.Circle(self.x_goal_[0], self.map_height_-self.x_goal_[1], radius=self.goal_radius_, color=(92, 214, 118), batch=self.batch_, group=self.foreground_)

        ## Check if goal radius was reached
        if self.nodes_distance(x_new, self.x_goal_) < self.goal_radius_ and self.node_to_cost_[x_new] < self.cost_to_goal_:
            print("Goal node radius reached!")

            print("x_new")
            print(x_new)
            print("x_new cost")
            print(self.node_to_cost_[x_new])

            self.cost_to_goal_ = self.node_to_cost_[x_new]

            ## Erase previous cost to goal text
            # self.path_cost_text_.text = 'Path cost: ' + str('%.2f' % self.cost_to_goal_)

            ## Draw cost to goal
            self.path_cost_text_.text = 'Path cost: ' + str('%.2f' % self.cost_to_goal_)
            # self.path_cost_text_ = pyglet.text.Label(
            #     'Path cost: ' + str('%.2f' % self.cost_to_goal_),
            #     font_name='Arial',
            #     font_size=self.font_size_, x=0, y=self.font_size_,
            #     batch=self.batch_, group=self.path_layer_)
            # self.path_cost_text_.add(pyglet.text.Label(
            #     'Path cost: ' + str('%.2f' % self.cost_to_goal_),
            #     font_name='Arial',
            #     font_size=self.font_size_, x=0, y=self.font_size_,
            #     batch=self.batch_, group=self.path_layer_))

            ## At least one path was found
            self.one_path_found_ = True

            ## Goal was found, get path to goal
            self.last_path_found_ = self.path(x_new, self.node_to_parent_)
            self.last_goal_point_ = x_new

        if self.one_path_found_:

            ## Clear last path
            self.path_line_ = set()

            ## Draw last path found
            for edge in self.last_path_found_:
                self.path_line_.add(
                    Path(edge[0][0], self.map_height_ - edge[0][1],
                         edge[1][0],
                         self.map_height_ - edge[1][1],
                         batch=self.batch_, group=self.path_layer_))

        self.batch_.draw()

        self.flip()

        if self.node_count_ == self.num_nodes_:
            print("Maximal number of nodes in tree reached!")
            print("Input anything and press enter to quit.")
            return False

        return self.path_found_

    def plan(self):
        ## Return true if path found, false otherwise
        path_found = False

        ## Count the number of nodes in tree
        node_count = 0

        ## Initialize nodes
        nodes = list()
        nodes.append(Node(self.x_init_, list(), 0, list(), 0))

        ## Initialize edges
        edges = set()

        ## Prepare tool to plot
        ## Get indices of coordinates with 1's in the drawing matrix
        map_height, map_width = self.scene_map_.shape
        screen_size = width, height = map_height, map_width
        pygame.font.init()
        myfont = pygame.font.SysFont('Arial', self.font_size_, bold=True)
        ## Black
        background_color = 0, 0, 0
        screen = pygame.display.set_mode(screen_size)
        screen.fill(background_color)
        ones_in_drawing = np.where(self.scene_map_ == 1)
        obstacles_coordinates = set(zip(ones_in_drawing[1], ones_in_drawing[0]))
        map_surface = pygame.image.load(self.map_name_).convert()
        input()

        ## Initialize path to goal empty
        path = list()

        while True:
            ## Sample point in space
            p_rand = self.sample_space(map_width, map_height)

            ## Check if point is inside any obstacle
            p_rand_tuple = (p_rand[0], p_rand[1])
            if p_rand_tuple in obstacles_coordinates:
                continue

            ## The first parent node is the initial node
            parent_node = nodes[0]

            ## Get nearest node to p_rand in the current tree
            for node in nodes:
                if self.nodes_distance(node.point(), p_rand) <= self.nodes_distance(parent_node.point(), p_rand):
                    parent_node = node

            ## Steer from nearest node in tree (i.e. parent_node) towards the
            ## p_rand to obtain a new node for the tree
            new_node_point = self.steer(parent_node.point(), p_rand, self.steer_delta_)

            ## Check if point is inside any obstacle
            new_node_point_tuple = (int(new_node_point[0]), int(new_node_point[1]))
            if new_node_point_tuple in obstacles_coordinates:
                continue

            ## Add a node to the tree
            node_count += 1

            ## New node childrens
            new_node_children = list()
            new_node_children.append(nodes[parent_node.index()].children())

            ## New node cost
            new_node_cost = self.nodes_distance(new_node_point, parent_node.point()) + parent_node.cost()

            ## Add new node
            nodes.append(Node(new_node_point, parent_node, node_count, new_node_children, new_node_cost))

            ## Add edge between parent and new node
            parent_node_tuple = (parent_node.point()[0], parent_node.point()[1])
            new_node_point_tuple = (new_node_point[0], new_node_point[1])
            edges.add((parent_node_tuple, new_node_point_tuple))

            ## Check if goal radius was reached
            if self.nodes_distance(new_node_point,
                                   self.x_goal_) < self.goal_radius_:
                print("Goal node radius reached!")

                # Store goal node
                goal_node = nodes[len(nodes)-1]

                ## Goal was found, get path to goal
                path = self.path(goal_node)

                path_found = True

            ## Draw everything
            self.draw(screen, self.x_init_, self.x_goal_, path, edges,
                      map_surface, new_node_cost, myfont, path_found)

            if path_found:
                input()
                break

            if node_count == self.num_nodes_:
                print("Maximal number of nodes in tree reached!")
                break

        return path_found

    def sample_space(self, x_max, y_max):
        x = random.randint(0, x_max)
        y = random.randint(0, y_max)

        x_rand = (x, y)

        return x_rand

    def nodes_distance(self, p1, p2):
        p1 = np.array([p1[0], p1[1]])
        p2 = np.array([p2[0], p2[1]])
        # diffnodes = p2 - p1
        # distance = np.sqrt(diffnodes[0]**2+diffnodes[1]**2)
        distance = np.linalg.norm(p1 - p2)

        return distance

    def steer(self, p1, p2, delta):
        p1 = np.array([p1[0], p1[1]])
        p2 = np.array([p2[0], p2[1]])
        if self.nodes_distance(p1, p2) < delta:
            p = p2
        else:
            diffnodes = p2 - p1
            diffnodes = diffnodes/self.nodes_distance(p1, p2)
            p = p1 + delta*diffnodes

        p = (p[0], p[1])

        return  p

    def draw(self, screen, x_init, x_goal, path, edges, map_surface, cost, myfont, path_found):

        pygame.event.get()

        ## Clear screen
        background_color = 19, 30, 31
        screen.fill(background_color)

        ## Draw scene map
        screen.blit(map_surface, (0, 0))

        ## Draw graph
        line_color = 196, 0, 183
        for edge in edges:
            line = pygame.draw.line(screen, line_color, edge[0],
                                    edge[1], width=2)

        ## Draw a circle around x_init
        init_color = 255, 207, 88
        pygame.draw.circle(screen, init_color, x_init, 10)

        ## Draw a circle of radius goal_radius around x_goal
        goal_color = 92, 214, 118
        pygame.draw.circle(screen, goal_color, x_goal,
                           self.goal_radius_)

        ## Keep drawing of the last path
        for node in path:
            path_color = 53, 155, 192
            pygame.draw.line(screen, path_color, node[0], node[1], width=5)

        ## Write cost on screen
        if path_found:
            textsurface = myfont.render('Path cost '+str('%.2f'%cost), False, (255, 255, 255))
            map_height, map_width = self.scene_map_.shape
            screen.blit(textsurface, (0, map_height - 50))

        pygame.display.flip()

    def path(self, goal_node, node_to_parent):
        path = list()

        ## Current node starts as the last node of the trajectory
        current_node = copy.deepcopy(goal_node)

        while True:
            ## Draw line from current node to current node parent
            path.append((current_node, node_to_parent[current_node]))
            # self.path_line_.add(Path(current_node[0],self.map_height_-current_node[1],node_to_parent[current_node][0],self.map_height_-node_to_parent[current_node][1], batch=self.batch_, group=self.foreground_))

            # draw_path_lines = shapes.Line(current_node[0], self.map_height_-current_node[1], node_to_parent[current_node][0], self.map_height_-node_to_parent[current_node][1], color=(53, 155, 192), width=4,
            #             batch=self.batch_, group=self.foreground_)

            ## Current node become its parent, we are searching for a path
            ## backwards in the tree
            parent_node = node_to_parent[current_node]
            current_node = parent_node

            print(parent_node)
            print(current_node)
            print(self.x_init_)

            ## If x_init is reached
            if sorted(current_node) == sorted(self.x_init_):
                break


        return path

    def nearest(self, x, x_init, rrt_graph):
        # ## The first nearest node is the initial node
        # nearest = x_init
        #
        # ## Get nearest node to x
        # for node in rrt_graph[0]:
        #     if self.nodes_distance(x, node) < self.nodes_distance(nearest, x):
        #         nearest = node

        nearest = list(rrt_graph[0].nearest(x, num_results=1, objects="raw"))
        nearest = nearest[0]

        return nearest


