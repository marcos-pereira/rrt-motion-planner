# Docker — rrt-motion-planner

## Image Layers

| # | Dockerfile | Image | Base | Content |
|---|-----------|-------|------|---------|
| 1 | `Dockerfile` | `rrt-motion-planner:latest` | `python:3.12-slim` | System libs (X11, OpenGL, FreeType) + all Python deps |
| 2 | `Dockerfile.webvis` | `rrt-motion-planner-webvis:latest` | layer 1 | Working dir `/app` + uvicorn entrypoint |

The webvis layer inherits FastAPI and uvicorn from the base image (they are in `python-scripts/requirements.txt`).

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Host (Fedora Linux)                    │
│                                                          │
│  X11 display (:0)          Browser                       │
│       │                       │ http://localhost:8000    │
│  ┌────┴──────────────┐  ┌─────┴──────────────────────┐  │
│  │   rrt-planner /   │  │       rrt-webvis           │  │
│  │   plan-then-draw  │  │  uvicorn server:app        │  │
│  │   pyglet → X11    │  │  FastAPI + PixiJS          │  │
│  └───────────────────┘  └────────────────────────────┘  │
│                                                          │
│  ../python-scripts/ ─(volume)─► /workspace (both)       │
│  ../webvis/         ─(volume)─► /app       (webvis only) │
└──────────────────────────────────────────────────────────┘
```

Both services mount `python-scripts/` as a live volume — code changes are reflected immediately without rebuilding.

## Quick Start

```bash
# 1 — Allow containers to connect to the host X server (desktop services only)
xhost +local:docker

# 2 — Copy the parameter template (edit docker/.env to change map, start/goal, budgets, ...)
cp docker/.env.example docker/.env

# 3 — Build images in order (webvis layer depends on the base)
docker compose -f docker/docker-compose.yml build rrt-planner
docker compose -f docker/docker-compose.yml build rrt-webvis

# 4a — Run the desktop visualizer
docker compose -f docker/docker-compose.yml --profile desktop up

# 4b — Run the web visualizer, then open http://localhost:8000
docker compose -f docker/docker-compose.yml --profile webvis up

# 4c — Run the differential-drive planner
docker compose -f docker/docker-compose.yml --profile differential-drive up

# 4d — Run the web visualizer for the differential-drive robot
docker compose -f docker/docker-compose.yml --profile webvis-differential-drive up
```

## Profiles

Each service has its own profile so only the requested one starts — profiles never interfere with each other.

| Profile | Service | Entry point | Description |
|---------|---------|------------|-------------|
| `desktop` | `rrt-planner` | `main.py` | RRT live step-by-step drawing, then RRT* continuous optimization |
| `plan-then-draw` | `plan-then-draw` | `plan_then_draw.py` | RRT runs fully first, then replays the tree edge-by-edge |
| `differential-drive` | `differential-drive` | `main_differential_drive.py` | Differential-drive robot: RRT then RRT* each plan fully, draw, and animate the robot along the found path |
| `webvis` | `rrt-webvis` | `server.py` (uvicorn) | PixiJS web visualizer — RRT at http://localhost:8000 |
| `webvis-rrtstar` | `rrt-webvis` | `server.py` (uvicorn) | Same server as `webvis` — RRT* at http://localhost:8000/rrtstar.html |
| `webvis-differential-drive` | `rrt-webvis` | `server.py` (uvicorn) | Same server as `webvis` — differential-drive RRT at http://localhost:8000/differential-drive.html |
| `webvis-differential-drive-rrtstar` | `rrt-webvis` | `server.py` (uvicorn) | Same server as `webvis` — differential-drive RRT* at http://localhost:8000/differential-drive-rrtstar.html |

## Examples

All commands from the project root.

```bash
# Desktop — default map (smile.png)
docker compose -f docker/docker-compose.yml --profile desktop up

# Desktop — plan-then-draw mode
docker compose -f docker/docker-compose.yml --profile plan-then-draw up

# Desktop — custom map and arguments
docker compose -f docker/docker-compose.yml --profile desktop run rrt-planner \
  python3 main.py maze1.png 15 10 50000 40 40 700 550

# Desktop — differential-drive robot, using parameters from docker/.env
docker compose -f docker/docker-compose.yml --profile differential-drive up

# Desktop — differential-drive robot, custom arguments instead of docker/.env
# (sampling_time is 20.0 rather than the 1.0 default — see the CLI Arguments note below)
docker compose -f docker/docker-compose.yml --profile differential-drive run differential-drive \
  python3 main_differential_drive.py smile.png 20 20000 30 30 30 460 20 8 20 1 20.0 10

# Web visualizer — RRT (open http://localhost:8000)
docker compose -f docker/docker-compose.yml --profile webvis up

# Web visualizer — RRT* (open http://localhost:8000/rrtstar.html)
# Uses the same server as the webvis profile; only one can run at a time.
docker compose -f docker/docker-compose.yml --profile webvis-rrtstar up

# Web visualizer — differential-drive robot, RRT (open http://localhost:8000/differential-drive.html)
docker compose -f docker/docker-compose.yml --profile webvis-differential-drive up

# Web visualizer — differential-drive robot, RRT* (open http://localhost:8000/differential-drive-rrtstar.html)
docker compose -f docker/docker-compose.yml --profile webvis-differential-drive-rrtstar up

# Shell inside the base container
docker compose -f docker/docker-compose.yml --profile desktop run rrt-planner bash
```

## CLI Arguments (`main.py`)

```
python3 main.py <map_name.png> <steer_step_size> <goal_radius> <max_nodes> <x_init> <y_init> <x_goal> <y_goal> [max_planning_time] [gamma_rrt] [eta_rrt] [near_radius]
```

| Argument | Example | Description |
|----------|---------|-------------|
| `map_name.png` | `smile.png` | Map image file (must be in `python-scripts/`) |
| `steer_step_size` | `15` | RRT steer step size in pixels |
| `goal_radius` | `10` | Goal ball radius in pixels |
| `max_nodes` | `50000` | Maximum nodes in the tree |
| `x_init` | `40` | Start x coordinate |
| `y_init` | `40` | Start y coordinate |
| `x_goal` | `700` | Goal x coordinate |
| `y_goal` | `550` | Goal y coordinate |
| `max_planning_time` | `30` | Optional maximum planning time in seconds. Omit for no time limit. |
| `gamma_rrt` | `1000` | Optional RRT* nearest-neighbor gain. Defaults to `1000`. |
| `eta_rrt` | `20` | Optional RRT* nearest-neighbor radius cap. Defaults to `20`. |
| `near_radius` | `20` | Optional RRT* `nearest_neighbor_radius` — accepted for backward compatibility but not actually used by the algorithm (see `RRTStar.__init__`'s docstring). Defaults to `20`. |

## CLI Arguments (`plan_then_draw.py`)

```
python3 plan_then_draw.py [map_name.png] [steer_step_size] [goal_radius] [max_nodes] [x_init] [y_init] [x_goal] [y_goal] [max_planning_time] [gamma_rrt] [eta_rrt] [near_radius]
```

Same arguments as `main.py` above, but every one of them is optional — any left out (or the whole command with no arguments at all) keeps its built-in default. Run `python3 plan_then_draw.py --help` for the defaults.

## CLI Arguments (`main_differential_drive.py`)

```
python3 main_differential_drive.py [map_name.png] [goal_radius] [max_nodes] [x_init] [y_init] [x_goal] [y_goal] [max_planning_time] [robot_radius] [linear_velocity_max] [angular_velocity_max] [sampling_time] [fps]
```

Plans for a `DifferentialDriveRobot` (state `[x, y, theta]`) instead of a point robot. Every argument is optional — any left out (or the whole command with no arguments at all) keeps its built-in default. Run `python3 main_differential_drive.py --help` for the defaults. RRT plans fully first; its tree and path are drawn, then a differential-drive robot animates along the found path at `fps` states per second (`PlanDrawer.animate_differential_drive_path()`). Press `Esc` in that window to close it and start planning RRT*, whose finished tree and path are drawn and animated the same way in a second window (press `Esc` there to close it).

Unlike `main.py`/`plan_then_draw.py`, tree expansion does not steer directly towards a sampled configuration: each RRT/RRT* iteration applies a randomly sampled `[linear_velocity, angular_velocity]` control to the nearest tree node for one simulated step (`DifferentialDriveRandomControlSteering`, built on `DifferentialDriveRobot`), following the kinodynamic RRT formulation in S. LaValle's *Planning Algorithms* (Section 5.3.1). The sampled configuration (drawn by `DifferentialDrivePoseSampler`) is only used to pick which existing tree node to extend from.

The robot itself is drawn as a circle (its footprint, radius `robot_radius`) with a heading line from the center to the edge in the direction of travel, plus a perpendicular axle line spanning the diameter, representing the wheel axle — both defined by `DifferentialDriveRobotShape` in `PlanDrawer.py`.

| Argument | Example | Description |
|----------|---------|-------------|
| `map_name.png` | `smile.png` | Map image file (must be in `python-scripts/`) |
| `goal_radius` | `20` | Goal ball radius in pixels |
| `max_nodes` | `20000` | Maximum nodes in the tree |
| `x_init` | `30` | Start x coordinate |
| `y_init` | `30` | Start y coordinate |
| `x_goal` | `30` | Goal x coordinate |
| `y_goal` | `460` | Goal y coordinate |
| `max_planning_time` | `20` | Optional maximum planning time in seconds |
| `robot_radius` | `8` | Circular footprint radius, used both by `DifferentialDriveCollisionChecker` and to draw the robot |
| `linear_velocity_max` | `20` | Upper bound of the sampled linear velocity (lower bound is always `0`) |
| `angular_velocity_max` | `1` | Upper bound of the sampled angular velocity, symmetric around `0` |
| `sampling_time` | `1.0` | Duration, in seconds, simulated per sampled control |
| `fps` | `10` | States of the final path drawn per second when animating the robot |

`sampling_time` also becomes `steer_delta` internally, which `DifferentialDriveSteering` uses directly as the distance (in pixels) driven per RRT/RRT* extension step. The default of `1.0` therefore advances only about a pixel per iteration — far too slow to converge across a map hundreds of pixels wide. Use a larger value such as `20.0` in practice.

Example command with every argument specified (matches the built-in defaults, except `sampling_time`, bumped to `20.0` per the note above):

```bash
python3 main_differential_drive.py smile.png 20 20000 30 30 30 460 20 8 20 1 20.0 10
```

## Configuring Planner Parameters (`.env`)

The `desktop` (`main.py`), `plan-then-draw` (`plan_then_draw.py`), and `differential-drive` (`main_differential_drive.py`) profiles read their CLI arguments from `docker/.env` — map, start/goal coordinates, node budget, max planning time, and each script's own tuning knobs — instead of hardcoded values in `docker-compose.yml`.

```bash
cp docker/.env.example docker/.env
# then edit docker/.env
```

Each variable also has a fallback default in `docker-compose.yml` matching `docker/.env.example`, so a missing file or a deleted line just falls back to that default rather than failing. See `docker/.env.example` for the full list of variables (prefixed `RRT_` for the `desktop` profile, `PLAN_THEN_DRAW_` for the `plan-then-draw` profile, `DIFF_DRIVE_` for the `differential-drive` profile).

`docker/.env` is gitignored since it's a local override; `docker/.env.example` is the tracked template.

## Rebuild After Dependency Changes

Only needed when `python-scripts/requirements.txt` changes:

```bash
docker compose -f docker/docker-compose.yml build rrt-planner
docker compose -f docker/docker-compose.yml build rrt-webvis
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DISPLAY` | `:0` | X11 display for GUI forwarding (desktop services) |
| `LIBGL_ALWAYS_SOFTWARE` | `1` | Forces Mesa software renderer (avoids DRI/DRM errors in Docker) |
| `MAPS_DIR` | `/workspace` | Path to `python-scripts/` inside the webvis container |

## GPU Hardware Rendering (optional)

By default the desktop containers use Mesa's software renderer (`LIBGL_ALWAYS_SOFTWARE=1`), which is sufficient for 2D visualization. To use the host GPU instead:

1. Set `LIBGL_ALWAYS_SOFTWARE=0` in `docker-compose.yml`
2. Add the DRI device passthrough under the desired service:

```yaml
devices:
  - /dev/dri:/dev/dri
```
