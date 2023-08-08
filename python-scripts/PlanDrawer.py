from pyglet import *
from pyglet.gl import *
import numpy as np

class Line():
    def __init__(self, x1, y1, x2, y2, batch, group):
        self.line_ = shapes.Line(x1, y1, x2, y2, color=(196, 0, 183), width=2, batch=batch, group=group)

class Path():
    def __init__(self, x1, y1, x2, y2, batch, group):
        self.path_ = shapes.Line(x1, y1, x2, y2, color=(53, 155, 192), width=4, batch=batch, group=group)

class PlanDrawer(pyglet.window.Window):
    def __init__(self,
                 map_name,
                 width,
                 height,
                 font_size,
                 *args,
                 **kwargs):
        super(PlanDrawer, self).__init__(width, height, *args,
                                         **kwargs)
        
        ## Pyglet
        self.batch_ = pyglet.graphics.Batch()
        self.background_ = pyglet.graphics.Group(order=0)
        self.foreground_ = pyglet.graphics.Group(order=1)
        self.image_layer_ = pyglet.graphics.Group(order=2)
        self.path_layer_ = pyglet.graphics.Group(order=3)
        
        ## Store lines
        self.lines_ = set()
        
        ## Store lines 
        self.lines_rrtstar_ = dict()

        ## Store path to draw
        self.path_line_ = set()
        
        ## Control window behavior
        self.drawing_ = 0
        self.stop_drawing_ = 0

        ## Load map
        self.map_ = image.load(map_name)
        self.map_sprite_ = pyglet.sprite.Sprite(self.map_, 0, 0,
                                                batch=self.batch_,
                                                group=self.image_layer_)
        
        self.map_height_ = height
        self.map_width_ = width
        
        self.font_size_ = font_size
        
        ## Store path cost text
        self.path_cost_text_ = pyglet.text.Label(
            'Path cost: ' + str(np.Inf),
            font_name='Arial',
            font_size=self.font_size_, x=0, y=self.font_size_,
            batch=self.batch_, group=self.path_layer_)

        self.key_ = pyglet.window.key
        
    def on_key_press(self, symbol, modifiers):
        """ Do event on key press from the symbol.

        Args:
            symbol (pyglet key_): the keyboard key
            modifiers (_type_): pyglet options (not being used now)
        """
        if symbol == self.key_.ESCAPE:
            self.drawing_ = 0
            self.stop_drawing_ = 1
            self.close()

        if symbol == self.key_.S:
            self.drawing_ = 1
        
    def draw_plan(self, planner):
        """ Returns True if plan still not found.

        Args:
            planner (RRTPlanner): The RRT planner.

        Returns:
            bool: True if plan still not found.
        """
        
        self.clear()
        
        # Clear last tree
        self.lines_ = set()
        
        plan_found, x_nearest, x_new = planner.run_step()
        
        plan_graph = planner.get_graph()
        for edge in plan_graph[1]:
            self.lines_.add(Line(edge[0][0], 
                            self.map_height_-edge[0][1], 
                            edge[1][0], 
                            self.map_height_-edge[1][1], 
                            self.batch_, 
                            self.foreground_))

        
        if plan_found:
            path = planner.path()
            print("plan found!")
            for node in path:
                self.path_line_.add(Path(node[0],
                                        self.map_height_-node[1],
                                        planner.node_to_parent_[node][0],
                                        self.map_height_-planner.node_to_parent_[node][1], 
                                        batch=self.batch_, 
                                        group=self.path_layer_))
    
        ## Draw x_init and x_goal
        draw_x_init = shapes.Circle(planner.x_init_[0], 
                                    self.map_height_-planner.x_init_[1], 
                                    radius=10, 
                                    color=(255, 207, 88), 
                                    batch=self.batch_, 
                                    group=self.foreground_)
        draw_x_goal = shapes.Circle(planner.x_goal_[0], 
                                    self.map_height_-planner.x_goal_[1], 
                                    radius=planner.goal_radius_, 
                                    color=(92, 214, 118), 
                                    batch=self.batch_, 
                                    group=self.foreground_)
        
        self.batch_.draw()
        
        # Ref: https://www.codingninjas.com/studio/library/the-application-event-loop-in-pyglet
        # Facilitates the dispatch of events
        self.flip()
        
        return not plan_found
            
    def run(self, planner):
        """ Run the planner and draw the planning after the key 's' is pressed.
        Press escape to stop.

        Args:
            planner (RRTPlanner): The RRT planner.
        """
        draw = True
        
        while True:
            if self.drawing_ == 1:        
                if draw:
                    draw = self.draw_plan(planner)
                
            if self.stop_drawing_ == 1:
                return

            event = self.dispatch_events()
            
        return
    
    def run_forever(self, planner):
        """ Run the planner and draw the planning after the key 's' is pressed.
        Press escape to stop.

        Args:
            planner (RRTPlanner): The RRT planner.
        """
        draw = True
        
        while True:
            if self.drawing_ == 1:        
                self.draw_plan_rrtstar(planner)
                
            if self.stop_drawing_ == 1:
                return

            self.dispatch_events()
            
        return

    def draw_plan_rrtstar(self, planner):
        """ Returns True if plan still not found.

        Args:
            planner (RRTPlanner): The RRT planner.

        Returns:
            bool: True if plan still not found.
        """
        
        self.clear()
                
        plan_found, x_nearest, x_new = planner.run_step()
        
        self.lines_rrtstar_[(planner.x_min_, x_new)] = \
            Line(planner.x_min_[0], 
                self.map_height_-planner.x_min_[1], 
                x_new[0], 
                self.map_height_-x_new[1], 
                self.batch_, 
                self.foreground_)
            
        for edge in planner.rewired_edges_:
            self.lines_rrtstar_[edge] = \
            Line(edge[0][0], 
                self.map_height_-edge[0][1], 
                edge[1][0], 
                self.map_height_-edge[1][1], 
                self.batch_, 
                self.foreground_)
        
        for edge in planner.remove_this_edges_:
            self.lines_rrtstar_.pop(edge)
        
        plan_graph = planner.get_graph()
        for edge in plan_graph[1]:
            self.lines_.add(Line(edge[0][0], 
                            self.map_height_-edge[0][1], 
                            edge[1][0], 
                            self.map_height_-edge[1][1], 
                            self.batch_, 
                            self.foreground_))

        
        if plan_found:
            path = planner.path()
            print("plan found!")
            for node in path:
                self.path_line_.add(Path(node[0],
                                        self.map_height_-node[1],
                                        planner.node_to_parent_[node][0],
                                        self.map_height_-planner.node_to_parent_[node][1], 
                                        batch=self.batch_, 
                                        group=self.path_layer_))
    
        ## Draw x_init and x_goal
        draw_x_init = shapes.Circle(planner.x_init_[0], 
                                    self.map_height_-planner.x_init_[1], 
                                    radius=10, 
                                    color=(255, 207, 88), 
                                    batch=self.batch_, 
                                    group=self.foreground_)
        draw_x_goal = shapes.Circle(planner.x_goal_[0], 
                                    self.map_height_-planner.x_goal_[1], 
                                    radius=planner.goal_radius_, 
                                    color=(92, 214, 118), 
                                    batch=self.batch_, 
                                    group=self.foreground_)
        
        self.batch_.draw()
        
        # Ref: https://www.codingninjas.com/studio/library/the-application-event-loop-in-pyglet
        # Facilitates the dispatch of events
        self.flip()
        
        return not plan_found
        