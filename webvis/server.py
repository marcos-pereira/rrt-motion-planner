import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

# Allow overriding the maps/scripts directory via env var for Docker
MAPS_DIR = Path(os.environ.get("MAPS_DIR", Path(__file__).parent.parent / "python-scripts")).resolve()
sys.path.insert(0, str(MAPS_DIR))

from Map import load_map
from RRT import RRT

app = FastAPI(title="RRT Web Visualizer")


@app.get("/maps-list")
def list_maps():
    """Return names of available PNG map files."""
    skip = {"no_background.png", "maze_no_background.png"}
    return sorted(f.name for f in MAPS_DIR.glob("*.png") if f.name not in skip)


@app.get("/plan")
def compute_plan(
    map_name: str = "smile.png",
    steer_delta: float = 15.0,
    goal_radius: int = 10,
    num_nodes: int = 20000,
    x0: int = 40,
    y0: int = 40,
    xg: int = 700,
    yg: int = 550,
):
    """Run the RRT planner and return edges in insertion order plus the path."""
    map_path = (MAPS_DIR / map_name).resolve()
    if map_path.parent != MAPS_DIR or map_path.suffix.lower() != ".png" or not map_path.is_file():
        raise HTTPException(status_code=404, detail=f"Map '{map_name}' not found.")

    # load_map uses relative paths and saves no_background.png to CWD
    original_dir = os.getcwd()
    os.chdir(MAPS_DIR)
    try:
        scene_map = load_map(map_name, test=True)
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


# Serve map PNG images — must be declared before the catch-all static mount
app.mount("/maps", StaticFiles(directory=str(MAPS_DIR)), name="maps")
# Serve the PixiJS frontend
app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static")
