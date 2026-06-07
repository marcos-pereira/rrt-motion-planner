# rrt-motion-planner

Implementations of RRT and RRT* sampling-based motion planners in Python, with both a desktop (pyglet) and a browser-based (PixiJS) visualization.

## Quick test with browser visualizer using docker

```bash
# 1 — Build images (webvis layer depends on the base image)
docker compose -f docker/docker-compose.yml build rrt-planner rrt-webvis

# 2a — Run the RRT web visualizer, then open http://localhost:8000
docker compose -f docker/docker-compose.yml --profile webvis up

# 2b — Run the RRT* web visualizer, then open http://localhost:8000/rrtstar.html
docker compose -f docker/docker-compose.yml --profile webvis-rrtstar up
```

---

## Quick view of the web visualizer

![webvisualizer_rrt](images/rrt.gif)

A full video of the Desktop visualizer is available [here](https://youtu.be/lUQy1XxTALI).

---

## Setup

Create and activate a virtual environment, then install all dependencies:

```bash
virtualenv my_venv
source my_venv/bin/activate
pip install -r python-scripts/requirements.txt
```

---

## Running the desktop visualizer

```bash
cd python-scripts
python3 main.py <map> <steer_delta> <goal_radius> <max_nodes> <x0> <y0> <xg> <yg>
```

| Argument | Description |
|---|---|
| `map` | PNG map file in `python-scripts/` |
| `steer_delta` | Steer step size in pixels |
| `goal_radius` | Goal ball radius in pixels |
| `max_nodes` | Maximum tree nodes |
| `x0 y0` | Start coordinates |
| `xg yg` | Goal coordinates |

Example commands:

```bash
python3 main.py smile.png      15 10  50000  40  40  700 550
python3 main.py simplemaze.png 15 10 100000  40  40  825 825
python3 main.py maze1.png      15 10 100000  40  40  750 750
```

A window with the loaded map opens first — close it to continue. A black pyglet window will open. Press `s` to start planning with RRT. Press `Esc` to close and open the RRT* window. Press `s` again to start RRT*.

---

## Running the web visualizer

The web visualizer runs in a browser using PixiJS. A single server serves both algorithms.

```bash
# From the project root
uvicorn webvis.server:app --host 0.0.0.0 --port 8000 --reload
```

**RRT** — open **http://localhost:8000**. Select a map, set the parameters, adjust the animation speed, and click **Run RRT**. The full tree is computed first, then animated edge-by-edge.

**RRT*** — open **http://localhost:8000/rrtstar.html**. Planning and drawing are interleaved: the tree is redrawn from scratch on every update because rewiring changes parent pointers. Use the **Steps per update** slider to balance responsiveness against rendering cost, and click **Stop** at any time.

---

## Running with Docker

See [`docker/README.md`](docker/README.md) for full instructions. Quick start:

```bash
# Allow the containers to connect to the host X server (desktop only)
xhost +local:docker

# Build images in order (webvis layer depends on the base)
docker compose -f docker/docker-compose.yml build rrt-planner
docker compose -f docker/docker-compose.yml build rrt-webvis

# Desktop visualizer
docker compose -f docker/docker-compose.yml --profile desktop up

# Plan-then-draw mode
docker compose -f docker/docker-compose.yml --profile plan-then-draw up

# Web visualizer (RRT) — open http://localhost:8000
docker compose -f docker/docker-compose.yml --profile webvis up

# Web visualizer (RRT*) — open http://localhost:8000/rrtstar.html
docker compose -f docker/docker-compose.yml --profile webvis-rrtstar up
```

---

## Adding new maps

The map must be a black-and-white PNG with obstacles in black.
