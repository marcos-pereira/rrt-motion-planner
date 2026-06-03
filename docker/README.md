# Docker — rrt-motion-planner

## Architecture

```
┌──────────────────────────────────────────┐
│              Host (Fedora Linux)          │
│                                          │
│  X11 display (:0)                        │
│       │                                  │
│  ┌────┴─────────────────────────────┐    │
│  │        rrt-motion-planner        │    │
│  │  python3 main.py <args>          │    │
│  │  pyglet window → X11 forwarded   │    │
│  └──────────────────────────────────┘    │
│                                          │
│  ../python-scripts/ ──(volume)──► /workspace │
└──────────────────────────────────────────┘
```

Source code in `python-scripts/` is mounted as a live volume — edits are reflected immediately on the next run, no rebuild needed.

## Quick Start

```bash
# 1 — Allow the container to connect to the host X server
xhost +local:docker

# 2 — Build the image (from the project root)
docker compose -f docker/docker-compose.yml build

# 3 — Run with default arguments
docker compose -f docker/docker-compose.yml up
```

## Examples

All commands from the project root.

```bash
# Run with default map (smile.png)
docker compose -f docker/docker-compose.yml up

# Run with a custom map and arguments
docker compose -f docker/docker-compose.yml run rrt-planner \
  python3 main.py maze1.png 15 10 50000 40 40 700 550

# Open a shell inside the container
docker compose -f docker/docker-compose.yml run rrt-planner bash
```

## CLI Arguments

```
python3 main.py <map_name.png> <steer_step_size> <goal_radius> <max_nodes> <x_init> <y_init> <x_goal> <y_goal>
```

| Argument | Example | Description |
|----------|---------|-------------|
| `map_name.png` | `smile.png` | Map image file (must be in `python-scripts/`) |
| `steer_step_size` | `15` | RRT steer step size in pixels |
| `goal_radius` | `10` | Goal region radius in pixels |
| `max_nodes` | `50000` | Maximum nodes in the tree |
| `x_init` | `40` | Start x coordinate |
| `y_init` | `40` | Start y coordinate |
| `x_goal` | `700` | Goal x coordinate |
| `y_goal` | `550` | Goal y coordinate |

## Rebuild After Dependency Changes

Only needed when `python-scripts/requirements.txt` changes:

```bash
docker compose -f docker/docker-compose.yml build
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DISPLAY` | `:0` | X11 display for GUI forwarding |
| `LIBGL_ALWAYS_SOFTWARE` | `1` | Forces Mesa software renderer (avoids DRI/DRM errors in Docker) |

## GPU Hardware Rendering (optional)

By default the container uses Mesa's software renderer (`LIBGL_ALWAYS_SOFTWARE=1`), which is sufficient for 2D visualization. To use the host GPU instead:

1. Set `LIBGL_ALWAYS_SOFTWARE=0` in `docker-compose.yml`
2. Add the DRI device passthrough under the `rrt-planner` service:

```yaml
devices:
  - /dev/dri:/dev/dri
```
