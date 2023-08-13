#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.
  
#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors: 
# marcos-pereira (https://github.com/marcos-pereira)

from pyglet import *
from pyglet.gl import *
import numpy as np

class Line():
    def __init__(self, x1, y1, x2, y2, batch, group):
        self.line_ = shapes.Line(x1, y1, x2, y2, color=(196, 0, 183), width=2, batch=batch, group=group)

class Path():
    def __init__(self, x1, y1, x2, y2, batch, group):
        self.path_ = shapes.Line(x1, y1, x2, y2, color=(11, 39, 219), width=5, batch=batch, group=group)

class PlanDrawer(pyglet.window.Window):
    def __init__(self,
                 map_name,
                 width,
                 height,
                 font_size,
                 *args,
                 **kwargs):
        """Return an object that draw the plans by executing the methods run() to stop at
        the first path to goal found or run_forever() to keep optimizing the path to goal
        in the case the RRTStar is used.

        Args:
            map_name (_type_): the name of the map figure <map_name>.png file. The
            .png extension is needed.
            width (_type_): the map width.
            height (_type_): the map height.
            font_size (_type_): the size of the font used to print the cost to goal of the path
            found in the map.
        """
        super(PlanDrawer, self).__init__(width, height, *args,
                                         **kwargs)
        
        ## Pyglet
        self.batch_ = pyglet.graphics.Batch()
        self.background_ = pyglet.graphics.Group(order=0)
        self.foreground_ = pyglet.graphics.Group(order=1)
        self.image_layer_ = pyglet.graphics.Group(order=2)
        self.path_layer_ = pyglet.graphics.Group(order=3)
        
        ## Store lines
        self.lines_ = list()
        
        ## Store lines 
        self.lines_rrtstar_ = dict()

        ## Store path to draw
        self.path_line_ = set()
        
        ## Path
        self.path_ = set()
        
        ## Last found path cost
        self.last_path_cost_ = np.inf
        
        ## Control window behavior
        self.drawing_ = 0
        self.stop_drawing_ = 0

        ## Load map
        self.map_ = image.load("no_background.png")
        self.map_sprite_ = pyglet.sprite.Sprite(self.map_, 0, 0,
                                                batch=self.batch_,
                                                group=self.background_)
        
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
        """ Do event on key press from the symbol. Press 'esc' to stop drawing and
        close window and press 's' to start planning and drawing.

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
        # self.lines_ = set()
        
        plan_found, x_nearest, x_new = planner.run_step()
        
        self.lines_.append(Line(x_nearest[0], 
                            self.map_height_-x_nearest[1], 
                            x_new[0], 
                            self.map_height_-x_new[1], 
                            self.batch_, 
                            self.foreground_))

        
        if plan_found:
            path, path_cost = planner.path(x_new)
            for i in range(len(path)-1):
                self.path_line_.add(Path(path[i][0],
                                        self.map_height_-path[i][1],
                                        path[i+1][0],
                                        self.map_height_-path[i+1][1], 
                                        batch=self.batch_, 
                                        group=self.path_layer_))
                
            ## Store path cost text
            self.path_cost_text_ = pyglet.text.Label(
            'Path cost: ' + str(path_cost),
            font_name='Arial',
            font_size=self.font_size_, x=0, y=self.font_size_,
            batch=self.batch_, group=self.path_layer_)
    
        ## Draw x_init and x_goal
        draw_x_init = shapes.Circle(planner.x_init_[0], 
                                    self.map_height_-planner.x_init_[1], 
                                    radius=planner.goal_radius_, 
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
        
        for edge in planner.remove_this_edges_:
            self.lines_rrtstar_.pop(edge)
            
        for edge in planner.rewired_edges_:
            self.lines_rrtstar_[edge] = \
            Line(edge[0][0], 
                self.map_height_-edge[0][1], 
                edge[1][0], 
                self.map_height_-edge[1][1], 
                self.batch_, 
                self.foreground_)
        
        self.lines_rrtstar_[(planner.x_min_, x_new)] = \
            Line(planner.x_min_[0], 
                self.map_height_-planner.x_min_[1], 
                x_new[0], 
                self.map_height_-x_new[1], 
                self.batch_, 
                self.foreground_)
            
        # plan_graph = planner.get_graph()
        # for edge in plan_graph[1]:
        #     self.lines_.add(Line(edge[0][0], 
        #                     self.map_height_-edge[0][1], 
        #                     edge[1][0], 
        #                     self.map_height_-edge[1][1], 
        #                     self.batch_, 
        #                     self.foreground_))

        
        if plan_found:
            self.path_line_ = set()
            path, path_cost = planner.path(planner.last_goal_node_)
            
            if path_cost < self.last_path_cost_:
                self.path_ = path
                
                ## Store path cost text
                self.path_cost_text_ = pyglet.text.Label(
                'Path cost: ' + str(path_cost),
                font_name='Arial',
                font_size=self.font_size_, x=0, y=self.font_size_,
                batch=self.batch_, group=self.path_layer_)
                    
            self.last_path_cost_ = path_cost
        
        for i in range(len(self.path_)-1):
            self.path_line_.add(Path(self.path_[i][0],
                                    self.map_height_-self.path_[i][1],
                                    self.path_[i+1][0],
                                    self.map_height_-self.path_[i+1][1], 
                                    batch=self.batch_, 
                                    group=self.path_layer_))
    
        ## Draw x_init and x_goal
        draw_x_init = shapes.Circle(planner.x_init_[0], 
                                    self.map_height_-planner.x_init_[1], 
                                    radius=planner.goal_radius_, 
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
        