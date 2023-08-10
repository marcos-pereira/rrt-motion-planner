# rrt-motion-planner
This repository will contain the implementation of some RRT motion planners.

# How to run this code?
First, install pygame, rtree, and scikit, and pyglet:
```
python3 -m pip install -U pygame --user
pip install rtree
pip install -U scikit-learn
pip install pyglet --user
```

Next, run the main.py:
```
python3 main.py
```

Close the loaded map window. A black window will open. Press `s` to start planning with the RRT. Press `esc` to stop planner and close window. Next, a new window will open an the RRT* will be executed after pressing `s` again.