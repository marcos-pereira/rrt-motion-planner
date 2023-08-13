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

Next, run the main.py with one of the .png maps as follows:
```
python3 main.py smile.png 15 10 50000 40 40 400 300
python3 main.py simplemaze.png 15 10 100000 40 40 825 825
python3 main.py maze1.png 15 10 50000 40 40 750 750
```

Close the loaded map window. A black window will open. Press `s` to start planning with the RRT. Press `esc` to stop planner and close window. Next, a new window will open an the RRT* will be executed after pressing `s` again.

# Adding new maps
The map must be black and white figure with obstacles in black.