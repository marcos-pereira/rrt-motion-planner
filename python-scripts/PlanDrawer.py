#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.
  
#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors: 
# marcos-pereira (https://github.com/marcos-pereira)

import math
import time

import pyglet
from pyglet import shapes, image
import numpy as np

from State import State
from TreeBuilder import TreeBuilder
from RRTPlanner import RRTPlanner
from RRTStar import RRTStar

class Line():
    def __init__(self, x1, y1, x2, y2, batch, group):
        self.line_ = shapes.Line(x1, y1, x2, y2, color=(196, 0, 183), thickness=2, batch=batch, group=group)

class Path():
    def __init__(self, x1, y1, x2, y2, batch, group):
        self.path_ = shapes.Line(x1, y1, x2, y2, color=(11, 39, 219), thickness=5, batch=batch, group=group)

class DifferentialDriveRobotShape():
    """ Draws a differential drive robot footprint as a circle with two lines showing its
    orientation: a heading line from the center to the edge of the circle in the direction
    the robot is facing, and an axle line perpendicular to the heading spanning the circle's
    diameter, representing the wheel axle.

    x, y and theta are given in the original map image frame, where y increases downward;
    map_height flips y so the shape is placed correctly in pyglet's y-up window, the same
    convention used by Line and Path elsewhere in this file.
    """

    def __init__(self, x, y, theta, radius, map_height, batch, group,
                 body_color=(219, 84, 11), heading_color=(255, 255, 255), axle_color=(255, 255, 255)):
        self.radius_ = radius

        draw_x, draw_y = x, map_height - y
        heading_x, heading_y = self._heading_endpoint(x, y, theta, radius, map_height)
        axle_x1, axle_y1, axle_x2, axle_y2 = self._axle_endpoints(x, y, theta, radius, map_height)

        self.body_ = shapes.Circle(draw_x, draw_y, radius, color=body_color, batch=batch, group=group)
        self.heading_ = shapes.Line(draw_x, draw_y, heading_x, heading_y, thickness=2,
                                    color=heading_color, batch=batch, group=group)
        self.axle_ = shapes.Line(axle_x1, axle_y1, axle_x2, axle_y2, thickness=2,
                                 color=axle_color, batch=batch, group=group)

    @staticmethod
    def _heading_endpoint(x, y, theta, radius, map_height):
        """ Return the (drawing-frame) endpoint of the heading line, from the center
        towards the edge of the circle in the direction theta. """
        return x + radius * math.cos(theta), map_height - (y + radius * math.sin(theta))

    @staticmethod
    def _axle_endpoints(x, y, theta, radius, map_height):
        """ Return the (drawing-frame) endpoints of the axle line, perpendicular to theta
        and spanning the circle's diameter. """
        axle_dx = radius * math.cos(theta + math.pi / 2)
        axle_dy = radius * math.sin(theta + math.pi / 2)
        return (x + axle_dx, map_height - (y + axle_dy),
                x - axle_dx, map_height - (y - axle_dy))

    def update(self, x, y, theta, map_height):
        """ Move this shape to state [x, y, theta], reusing the same pyglet shapes instead
        of creating new ones, so animating a path does not accumulate shapes in the batch.
        """
        draw_x, draw_y = x, map_height - y
        self.body_.x, self.body_.y = draw_x, draw_y

        heading_x, heading_y = self._heading_endpoint(x, y, theta, self.radius_, map_height)
        self.heading_.x, self.heading_.y = draw_x, draw_y
        self.heading_.x2, self.heading_.y2 = heading_x, heading_y

        axle_x1, axle_y1, axle_x2, axle_y2 = self._axle_endpoints(x, y, theta, self.radius_, map_height)
        self.axle_.x, self.axle_.y = axle_x1, axle_y1
        self.axle_.x2, self.axle_.y2 = axle_x2, axle_y2

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
            'Path cost: ' + str(np.inf),
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
            
    def draw(self, tree_builder : TreeBuilder, state_goal : State, goal_radius : int, path : list[tuple[int, int]]):
        """Draw the nodes and edges in the graph.

        Args:
            tree_builder (TreeBuilder): the tree builder containing the graph information.
            state_goal (State): the goal configuration.
            edges_in_graph (list): the list of edges in the graph.
        """
        self.clear()

        goal_node = state_goal.get_value()

        # Get init node to start drawing the graph from the root node
        init_node = tree_builder.get_init_node()
        edges_in_order = tree_builder.get_edges_in_order()
        
        # Goal reached
        goal_reached = False
        goal_color = (92, 214, 118, 100)

        # Draw edges in the exact order state_new was generated.
        for parent_node, child_node in edges_in_order:
            self.lines_.append(Line(parent_node[0], 
                                    self.map_height_-parent_node[1], 
                                    child_node[0], 
                                    self.map_height_-child_node[1], 
                                    self.batch_, 
                                    self.foreground_))

            p_child = np.array(child_node)
            p_goal = np.array(goal_node)
            distance_to_goal = np.linalg.norm(p_child - p_goal)

            if distance_to_goal <= goal_radius:
                goal_color = (10, 214, 118)
                goal_reached = True

            # Draw x_init and x_goal
            draw_x_init = shapes.Circle(init_node[0], 
                                        self.map_height_-init_node[1], 
                                        radius=goal_radius, 
                                        color=(255, 207, 88), 
                                        batch=self.batch_, 
                                        group=self.foreground_)
            draw_x_goal = shapes.Circle(goal_node[0], 
                                        self.map_height_-goal_node[1], 
                                        radius=goal_radius, 
                                        color=goal_color, 
                                        batch=self.batch_, 
                                        group=self.foreground_)
            
            self.batch_.draw()
            
            # Ref: https://www.codingninjas.com/studio/library/the-application-event-loop-in-pyglet
            # Facilitates the dispatch of events
            self.flip()
            
            event = self.dispatch_events()

        # Draw path to goal if goal reached
        if goal_reached:
            for i in range(len(path)-1):
                self.path_line_.add(Path(path[i][0],
                                        self.map_height_-path[i][1],
                                        path[i+1][0],
                                        self.map_height_-path[i+1][1], 
                                        batch=self.batch_, 
                                        group=self.path_layer_))
            
            self.batch_.draw()
            
            # Ref: https://www.codingninjas.com/studio/library/the-application-event-loop-in-pyglet
            # Facilitates the dispatch of events
            self.flip()
        
    def draw_and_plan(self, planner : RRTPlanner):
        """ Returns True if drawing should continue, i.e. if the path is still not found
        and neither the max number of nodes nor the max planning time has been reached.

        Args:
            planner (RRTPlanner): The RRT planner.

        Returns:
            bool: True if drawing should continue.
        """
        
        self.clear()
        
        # Clear last tree
        # self.lines_ = set()
        
        plan_found, state_nearest, state_new = planner.run_step()

        # Draw tree
        tree_node = planner.get_tree_nodes()

        # Run the tree node map in reverse order to draw the edges from the root node to the new node added in the tree,
        # which is the opposite order of how the nodes were added to the tree node map,
        # since the tree node map is built in the order of how the nodes were added to the tree,
        # where each node has a pointer to its parent node.
        current_node = tree_node[-1]
        while current_node.get_parent() is not None:
            parent_node = current_node.get_parent()
            self.lines_.append(Line(parent_node.get_node_coordinates()[0],
                                    self.map_height_-parent_node.get_node_coordinates()[1],
                                    current_node.get_node_coordinates()[0],
                                    self.map_height_-current_node.get_node_coordinates()[1],
                                    self.batch_,
                                    self.foreground_))
            current_node = parent_node

        if plan_found:
            path, path_cost = planner.path(state_new)
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
    
        ## Draw state_init and state_goal
        state_init_value = planner.state_init_.get_value()
        state_goal_value = planner.state_goal_.get_value()
        draw_x_init = shapes.Circle(state_init_value[0],
                                    self.map_height_-state_init_value[1],
                                    radius=planner.goal_radius_,
                                    color=(255, 207, 88),
                                    batch=self.batch_,
                                    group=self.foreground_)
        draw_x_goal = shapes.Circle(state_goal_value[0],
                                    self.map_height_-state_goal_value[1],
                                    radius=planner.goal_radius_,
                                    color=(92, 214, 118),
                                    batch=self.batch_,
                                    group=self.foreground_)

        self.batch_.draw()

        # Ref: https://www.codingninjas.com/studio/library/the-application-event-loop-in-pyglet
        # Facilitates the dispatch of events
        self.flip()

        budget_exhausted = planner.max_number_nodes() or planner.max_planning_time_reached()

        return not (plan_found or budget_exhausted)

    def run(self, planner):
        """ Run the planner and draw the planning after the key 's' is pressed.
        Press escape to stop, or wait for the goal to be found, the max number of
        nodes, or the max planning time to be reached.

        Args:
            planner (RRTPlanner): The RRT planner.
        """
        draw = True
        timer_started = False

        while True:
            if self.drawing_ == 1:
                if not timer_started:
                    planner.start_planning_timer()
                    timer_started = True

                if draw:
                    draw = self.draw_and_plan(planner)

            if self.stop_drawing_ == 1:
                return

            event = self.dispatch_events()

        return
    
    def run_forever(self, planner):
        """ Run the planner and draw the planning after the key 's' is pressed, continuing
        to rewire and lower the path cost even after the first path is found. Press escape
        to stop, or wait for the max number of nodes or the max planning time to be reached.

        Args:
            planner (RRTPlanner): The RRT planner.
        """
        draw = True
        timer_started = False

        while True:
            if self.drawing_ == 1:
                if not timer_started:
                    planner.start_planning_timer()
                    timer_started = True

                if draw:
                    draw = self.draw_plan_rrtstar(planner)

            if self.stop_drawing_ == 1:
                return

            self.dispatch_events()

        return

    def draw_plan_rrtstar(self, planner : RRTStar):
        """ Returns True if drawing should continue, i.e. if neither the max number of
        nodes nor the max planning time has been reached. Unlike draw_and_plan(), finding
        a path does not stop the drawing, since RRT* keeps rewiring to lower its cost.

        Args:
            planner (RRTStar): The RRT* planner.

        Returns:
            bool: True if drawing should continue.
        """
        
        self.clear()
        
        # Clear last tree
        self.lines_ = list()
                        
        plan_found, state_nearest, state_new = planner.run_step()

        # Draw tree
        tree_node = planner.get_tree_nodes()

        # Draw edges for ALL nodes in the tree
        for node in tree_node:
            parent = node.get_parent()
            if parent is not None:
                self.lines_.append(Line(parent.get_node_coordinates()[0], 
                                        self.map_height_ - parent.get_node_coordinates()[1], 
                                        node.get_node_coordinates()[0], 
                                        self.map_height_ - node.get_node_coordinates()[1], 
                                        self.batch_, 
                                        self.foreground_))
        
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
    
        ## Draw state_init and state_goal
        state_init_value = planner.state_init_.get_value()
        state_goal_value = planner.state_goal_.get_value()
        draw_x_init = shapes.Circle(state_init_value[0],
                                    self.map_height_-state_init_value[1],
                                    radius=planner.goal_radius_,
                                    color=(255, 207, 88),
                                    batch=self.batch_,
                                    group=self.foreground_)
        draw_x_goal = shapes.Circle(state_goal_value[0],
                                    self.map_height_-state_goal_value[1],
                                    radius=planner.goal_radius_,
                                    color=(92, 214, 118),
                                    batch=self.batch_,
                                    group=self.foreground_)

        self.batch_.draw()

        # Ref: https://www.codingninjas.com/studio/library/the-application-event-loop-in-pyglet
        # Facilitates the dispatch of events
        self.flip()

        budget_exhausted = planner.max_number_nodes() or planner.max_planning_time_reached()

        return not budget_exhausted

    def animate_differential_drive_path(self, path: list[tuple[float, float, float]],
                                         robot_radius: float, fps: float):
        """ Animate a differential drive robot moving along path, drawing only the final
        path followed rather than the tree growth. Whatever is already on this window's
        batch (e.g. the tree and path drawn by draw() or draw_final()) stays visible as a
        static background, with the moving robot drawn on top of it.

        Args:
            path (list): chronological list of [x, y, theta] states, from state_init to
            state_goal, e.g. built by reversing and prepending state_init to the list
            returned by RRTPlanner.path()/plan() (which is ordered from state_goal back
            towards state_init, and excludes state_init itself).
            robot_radius (float): radius of the circle used to draw the robot footprint.
            fps (float): number of path states drawn per second.
        """
        if not path:
            return

        seconds_per_frame = 1.0 / fps
        robot_shape = None

        for x, y, theta in path:
            self.clear()

            if robot_shape is None:
                robot_shape = DifferentialDriveRobotShape(x, y, theta, robot_radius,
                                                          self.map_height_, self.batch_, self.path_layer_)
            else:
                robot_shape.update(x, y, theta, self.map_height_)

            self.batch_.draw()

            # Ref: https://www.codingninjas.com/studio/library/the-application-event-loop-in-pyglet
            # Facilitates the dispatch of events
            self.flip()
            self.dispatch_events()

            if self.stop_drawing_ == 1:
                return

            time.sleep(seconds_per_frame)

    def draw_final(self, planner : RRTPlanner, path : list[tuple[int, int]], path_cost : float):
        """ Draw the finished tree and path of a planner that has already been run to
        completion, e.g. via plan(). Unlike draw(), which replays the TreeBuilder's
        append-only edge log, this walks each node's current parent pointer, so it stays
        correct for RRT* trees whose edges get rewired after being first added.

        Args:
            planner (RRTPlanner): the RRT or RRT* planner, already run to completion.
            path (list): the path from state_init to state_goal, or an empty list if none was found.
            path_cost (float): the cost of path, as returned alongside it by plan().
        """
        self.clear()

        for node in planner.get_tree_nodes():
            parent = node.get_parent()
            if parent is not None:
                self.lines_.append(Line(parent.get_node_coordinates()[0],
                                        self.map_height_-parent.get_node_coordinates()[1],
                                        node.get_node_coordinates()[0],
                                        self.map_height_-node.get_node_coordinates()[1],
                                        self.batch_,
                                        self.foreground_))

        for i in range(len(path)-1):
            self.path_line_.add(Path(path[i][0],
                                    self.map_height_-path[i][1],
                                    path[i+1][0],
                                    self.map_height_-path[i+1][1],
                                    batch=self.batch_,
                                    group=self.path_layer_))

        if path:
            self.path_cost_text_ = pyglet.text.Label(
                'Path cost: ' + str(path_cost),
                font_name='Arial',
                font_size=self.font_size_, x=0, y=self.font_size_,
                batch=self.batch_, group=self.path_layer_)

        ## Draw state_init and state_goal
        state_init_value = planner.state_init_.get_value()
        state_goal_value = planner.state_goal_.get_value()
        draw_x_init = shapes.Circle(state_init_value[0],
                                    self.map_height_-state_init_value[1],
                                    radius=planner.goal_radius_,
                                    color=(255, 207, 88),
                                    batch=self.batch_,
                                    group=self.foreground_)
        draw_x_goal = shapes.Circle(state_goal_value[0],
                                    self.map_height_-state_goal_value[1],
                                    radius=planner.goal_radius_,
                                    color=(92, 214, 118),
                                    batch=self.batch_,
                                    group=self.foreground_)

        self.batch_.draw()

        # Ref: https://www.codingninjas.com/studio/library/the-application-event-loop-in-pyglet
        # Facilitates the dispatch of events
        self.flip()
