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
```

## Profiles

Each service has its own profile so only the requested one starts — profiles never interfere with each other.

| Profile | Service | Entry point | Description |
|---------|---------|------------|-------------|
| `desktop` | `rrt-planner` | `main.py` | RRT live step-by-step drawing, then RRT* continuous optimization |
| `plan-then-draw` | `plan-then-draw` | `plan_then_draw.py` | RRT runs fully first, then replays the tree edge-by-edge |
| `webvis` | `rrt-webvis` | `server.py` (uvicorn) | PixiJS web visualizer — RRT at http://localhost:8000 |
| `webvis-rrtstar` | `rrt-webvis` | `server.py` (uvicorn) | Same server as `webvis` — RRT* at http://localhost:8000/rrtstar.html |

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

# Web visualizer — RRT (open http://localhost:8000)
docker compose -f docker/docker-compose.yml --profile webvis up

# Web visualizer — RRT* (open http://localhost:8000/rrtstar.html)
# Uses the same server as the webvis profile; only one can run at a time.
docker compose -f docker/docker-compose.yml --profile webvis-rrtstar up

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

## Configuring Planner Parameters (`.env`)

The `desktop` (`main.py`) and `plan-then-draw` (`plan_then_draw.py`) profiles read their CLI arguments from `docker/.env` — map, start/goal coordinates, node budget, max planning time, and RRT* tuning (`gamma_rrt`, `eta_rrt`, `near_radius`) — instead of hardcoded values in `docker-compose.yml`.

```bash
cp docker/.env.example docker/.env
# then edit docker/.env
```

Each variable also has a fallback default in `docker-compose.yml` matching `docker/.env.example`, so a missing file or a deleted line just falls back to that default rather than failing. See `docker/.env.example` for the full list of variables (prefixed `RRT_` for the `desktop` profile, `PLAN_THEN_DRAW_` for the `plan-then-draw` profile).

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
