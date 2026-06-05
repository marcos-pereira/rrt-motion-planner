import json
import os
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Allow overriding the maps/scripts directory via env var for Docker
MAPS_DIR = Path(os.environ.get("MAPS_DIR", Path(__file__).parent.parent / "python-scripts")).resolve()
CWD_LOCK = threading.Lock()
sys.path.insert(0, str(MAPS_DIR))

from Map import load_map
from RRT import RRT
from RRTStar import RRTStar

app = FastAPI(title="RRT Web Visualizer")


@app.get("/maps-list")
def list_maps():
    """Return names of available PNG map files."""
    skip = {"no_background.png", "maze_no_background.png"}
    return sorted(f.name for f in MAPS_DIR.glob("*.png") if f.name not in skip)


@app.get("/maps/{map_name}")
def get_map(map_name: str):
    """Serve a single PNG map file without exposing the entire python-scripts/ directory."""
    map_path = (MAPS_DIR / map_name).resolve()
    if map_path.parent != MAPS_DIR or map_path.suffix.lower() != ".png" or not map_path.is_file():
        raise HTTPException(status_code=404, detail=f"Map '{map_name}' not found.")
    return FileResponse(map_path, media_type="image/png")

@app.get("/plan")
def compute_plan(
    map_name: str = "smile.png",
    steer_delta: float = Query(15.0, ge=1, le=500),
    goal_radius: int = Query(10, ge=1, le=500),
    num_nodes: int = Query(20000, ge=100, le=200000),
    x0: int = Query(40, ge=0),
    y0: int = Query(40, ge=0),
    xg: int = Query(700, ge=0),
    yg: int = Query(550, ge=0),
):
    """Run the RRT planner and return edges in insertion order plus the path."""
    map_path = (MAPS_DIR / map_name).resolve()
    if map_path.parent != MAPS_DIR or map_path.suffix.lower() != ".png" or not map_path.is_file():
        raise HTTPException(status_code=404, detail=f"Map '{map_name}' not found.")

    # load_map uses relative paths and saves no_background.png to CWD
    # (process-wide), so serialize planning runs.
    with CWD_LOCK:
        original_dir = os.getcwd()
        os.chdir(MAPS_DIR)
        try:
            try:
                scene_map = load_map(map_name, test=True)
            except Exception:
                raise HTTPException(status_code=400, detail=f"Failed to load map '{map_name}'.")
            map_height, map_width = scene_map.shape
            x_init = (x0, y0)
            x_goal = (xg, yg)

            rrt = RRT(x_init, x_goal, goal_radius, int(steer_delta), scene_map, num_nodes)

            path: list = []
            path_cost: float = float("inf")
            try:
                path, path_cost = rrt.run()
            except (NameError, UnboundLocalError):
                # Max nodes reached before a path was found
                pass

            edges = rrt.tree_builder_.get_edges_in_order()
        finally:
            os.chdir(original_dir)

    return {
        "edges": [[list(e[0]), list(e[1])] for e in edges],
        "path": [list(p) for p in path],
        "map_width": map_width,
        "map_height": map_height,
        "x_init": list(x_init),
        "x_goal": list(x_goal),
        "goal_radius": goal_radius,
        "path_cost": path_cost,
        "node_count": len(edges) + 1,
        "path_found": len(path) > 0,
        "map_name": map_name,
    }


@app.get("/plan-rrtstar")
def stream_rrtstar(
    map_name: str = "smile.png",
    steer_delta: float = Query(15.0, ge=1, le=500),
    goal_radius: int = Query(10, ge=1, le=500),
    num_nodes: int = Query(20000, ge=100, le=200000),
    x0: int = Query(40, ge=0),
    y0: int = Query(40, ge=0),
    xg: int = Query(700, ge=0),
    yg: int = Query(550, ge=0),
    batch_size: int = Query(100, ge=1, le=5000),
    gamma_rrt: float = Query(1000.0, ge=1.0),
    eta: float = Query(20.0, ge=1.0),
):
    """Run RRT* step-by-step and stream full tree snapshots as Server-Sent Events.

    Each event carries the complete current tree so the browser can clear and
    redraw on every snapshot — necessary because rewiring changes parent pointers.
    """
    map_path = (MAPS_DIR / map_name).resolve()
    if map_path.parent != MAPS_DIR or map_path.suffix.lower() != ".png" or not map_path.is_file():
        raise HTTPException(status_code=404, detail=f"Map '{map_name}' not found.")

    def generate():
        # Hold the CWD lock only for load_map (it writes no_background.png to CWD).
        # RRT* planning itself does not touch the filesystem.
        with CWD_LOCK:
            original_dir = os.getcwd()
            os.chdir(MAPS_DIR)
            try:
                try:
                    scene_map = load_map(map_name, test=True)
                except Exception as exc:
                    yield f"data: {json.dumps({'error': str(exc)})}\n\n"
                    return
                map_height, map_width = scene_map.shape
            finally:
                os.chdir(original_dir)

        x_init = (x0, y0)
        x_goal = (xg, yg)

        rrtstar = RRTStar(
            x_init, x_goal, goal_radius, int(steer_delta),
            eta, gamma_rrt,
            20,      # nearest_neighbor_radius — unused per docstring
            scene_map, num_nodes,
        )

        step = 0
        while True:
            try:
                rrtstar.run_step()
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
                return

            step += 1
            done = bool(rrtstar.max_number_nodes())

            if step % batch_size == 0 or done:
                # Build edge list from the current parent-pointer tree.
                # Rewiring updates parent pointers, so this always reflects the
                # latest tree structure.
                edges = [
                    [list(node.get_parent().get_node_coordinates()),
                     list(node.get_node_coordinates())]
                    for node in rrtstar.tree_nodes_
                    if node.get_parent() is not None
                ]

                path_nodes: list = []
                path_cost = None
                if rrtstar.last_goal_node_ is not None:
                    path_list, cost = rrtstar.path(rrtstar.last_goal_node_)
                    path_nodes = [list(p) for p in path_list]
                    path_cost = float(cost)

                snapshot = {
                    "edges": edges,
                    "path": path_nodes,
                    "path_cost": path_cost,
                    "path_found": rrtstar.last_goal_node_ is not None,
                    "node_count": rrtstar.node_count_,
                    "map_width": int(map_width),
                    "map_height": int(map_height),
                    "x_init": list(x_init),
                    "x_goal": list(x_goal),
                    "goal_radius": goal_radius,
                    "map_name": map_name,
                    "done": done,
                }

                yield f"data: {json.dumps(snapshot)}\n\n"

            if done:
                break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serve the PixiJS frontend
app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static")
