import json
import os
import sys
import threading
import time
from pathlib import Path

import anyio.from_thread
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Allow overriding the maps/scripts directory via env var for Docker
MAPS_DIR = Path(os.environ.get("MAPS_DIR", Path(__file__).parent.parent / "python-scripts")).resolve()
# Limits the four planning endpoints to one run at a time across all clients, so a
# handful of browser tabs can't each kick off a full RRT/RRT* run and exhaust the
# server's CPU.
PLANNING_SEMAPHORE = threading.Semaphore(1)
sys.path.insert(0, str(MAPS_DIR))

from Map import load_map
from RRT import RRT
from RRTStar import RRTStar
from SimpleDeltaSteering import SimpleDeltaSteering
from Cartesian2DSampler import Cartesian2DSampler
from Cartesian2DCollisionChecker import Cartesian2DCollisionChecker
from DifferentialDrivePoseSampler import DifferentialDrivePoseSampler
from DifferentialDriveSteering import DifferentialDriveSteering
from DifferentialDriveCollisionChecker import DifferentialDriveCollisionChecker
from RealVectorState import RealVectorState

app = FastAPI(title="RRT Web Visualizer")


@app.middleware("http")
async def no_cache_static_assets(request, call_next):
    """Force browsers to revalidate static/ files (HTML/JS/CSS) on every request
    instead of serving a stale cached copy after a deploy. Revalidation still lets
    the browser skip the download via a 304 when the file is unchanged (StaticFiles
    handles ETag/If-None-Match), so this doesn't disable caching outright — it just
    prevents an old app.js from being used silently without a hard refresh."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"
    return response


def _resolve_map_path(map_name: str) -> Path:
    """Validate map_name and return its path, without exposing the rest of MAPS_DIR."""
    map_path = (MAPS_DIR / map_name).resolve()
    if map_path.parent != MAPS_DIR or map_path.suffix.lower() != ".png" or not map_path.is_file():
        raise HTTPException(status_code=404, detail=f"Map '{map_name}' not found.")
    return map_path


def _acquire_planning_slot():
    """Reserve the single planning slot, or raise 429 immediately if one is already
    in use. Rejecting outright (rather than queuing) matters most for the streaming
    endpoints: a client waiting on an SSE response would otherwise see nothing at
    all until the earlier run's deadline elapses."""
    if not PLANNING_SEMAPHORE.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="A planning job is already running. Please wait for it to finish, or stop it, before starting another.",
        )


def _disconnect_checker(request: Request):
    """Return a callable, safe to call from the worker thread a streaming
    generator runs in, that reports whether request's client has gone away.

    anyio.from_thread.run() hands the coroutine back to this request's event loop
    and blocks the calling thread until it resolves; this works here because
    Starlette drives a sync generator's next() calls via anyio.to_thread.run_sync(),
    which sets up exactly the thread/event-loop pairing anyio.from_thread.run()
    needs.
    """
    return lambda: anyio.from_thread.run(request.is_disconnected)


def _load_scene_map(map_name: str):
    """Load a map's occupancy grid, returning (scene_map, map_width, map_height)."""
    scene_map = load_map(map_name, test=True, maps_dir=MAPS_DIR)
    map_height, map_width = scene_map.shape
    return scene_map, map_width, map_height


def _run_to_completion(planner, max_planning_time):
    """Drive planner.run_step() to completion, returning (path, path_cost, stop_reason).

    Stops at the first path found, when the node budget is reached, or when
    max_planning_time elapses — enforced here via a wall-clock check after every
    single step, rather than relying on the planner's own internal timer (which only
    gets a chance to fire between calls to run_step()).
    """
    deadline = (time.monotonic() + max_planning_time) if max_planning_time is not None else None
    path, path_cost, stop_reason = [], float("inf"), "max_nodes"

    while True:
        path_found_step, state_nearest, state_new = planner.run_step()

        if path_found_step:
            path, path_cost = planner.path(state_new)
            stop_reason = "goal_reached"
            break

        if planner.node_count_ >= planner.max_num_nodes_:
            stop_reason = "max_nodes"
            break

        if deadline is not None and time.monotonic() >= deadline:
            stop_reason = "max_time"
            break

    return path, path_cost, stop_reason


def _to_chronological_path(path, x_init):
    """Reverse an RRTPlanner.path()/path_list result (ordered from the goal back
    towards state_init, excluding state_init) into chronological order from
    state_init to goal, with state_init prepended, so the frontend can animate a
    differential-drive robot along it directly, start to goal."""
    return [list(x_init.get_value())] + [list(p) for p in reversed(path)]


def _rrtstar_edges(rrtstar):
    """Build an edge list from RRT*'s current parent-pointer tree. Rewiring updates
    parent pointers, so this always reflects the latest tree structure."""
    return [
        [list(node.get_parent().get_state().get_value()),
         list(node.get_state().get_value())]
        for node in rrtstar.tree_nodes_
        if node.get_parent() is not None
    ]


def _stream_rrtstar_events(rrtstar, batch_size, max_planning_time, snapshot_extra, path_builder, is_disconnected):
    """Drive rrtstar.run_step(), yielding a JSON SSE snapshot every batch_size steps
    and once more on completion. Each snapshot carries the complete current tree so
    the browser can clear and redraw on every update — necessary because rewiring
    changes parent pointers. snapshot_extra is merged into every snapshot as-is
    (map/robot metadata that doesn't change during planning); path_builder converts
    a raw path_list into the "path" field.

    is_disconnected is checked once per batch and, if the client is gone, this
    returns instead of continuing to plan. This is the only reliable way to stop:
    once the client's socket is closed, Starlette detects it only on the next
    failed send() and then simply stops calling next() on this generator, without
    ever closing it — so without this check the generator (and the caller's
    "finally" that releases PLANNING_SEMAPHORE) would never run again, wedging the
    planning slot until the server restarts.
    """
    deadline = (time.monotonic() + max_planning_time) if max_planning_time is not None else None

    step = 0
    while True:
        try:
            rrtstar.run_step()
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            return

        step += 1
        node_budget_reached = rrtstar.node_count_ >= rrtstar.max_num_nodes_
        time_budget_reached = deadline is not None and time.monotonic() >= deadline
        done = node_budget_reached or time_budget_reached
        stop_reason = ("max_nodes" if node_budget_reached else "max_time") if done else None

        if step % batch_size == 0 or done:
            if is_disconnected():
                return

            path_nodes: list = []
            path_cost = None
            if rrtstar.last_goal_node_ is not None:
                path_list, cost = rrtstar.path(rrtstar.last_goal_node_)
                path_nodes = path_builder(path_list)
                path_cost = float(cost)

            snapshot = {
                "edges": _rrtstar_edges(rrtstar),
                "path": path_nodes,
                "path_cost": path_cost,
                "path_found": rrtstar.last_goal_node_ is not None,
                "node_count": rrtstar.node_count_,
                "done": done,
                "stop_reason": stop_reason,
                **snapshot_extra,
            }

            yield f"data: {json.dumps(snapshot)}\n\n"

        if done:
            break


@app.get("/maps-list")
def list_maps():
    """Return names of available PNG map files."""
    skip = {"no_background.png", "maze_no_background.png"}
    return sorted(f.name for f in MAPS_DIR.glob("*.png") if f.name not in skip)


@app.get("/maps/{map_name}")
def get_map(map_name: str):
    """Serve a single PNG map file without exposing the entire python-scripts/ directory."""
    map_path = _resolve_map_path(map_name)
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
    max_planning_time: float | None = Query(None, ge=0.1, le=300),
):
    """Run the RRT planner and return edges in insertion order plus the path."""
    _resolve_map_path(map_name)
    _acquire_planning_slot()
    try:
        try:
            scene_map, map_width, map_height = _load_scene_map(map_name)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Failed to load map '{map_name}'.")

        x_init = RealVectorState((x0, y0))
        x_goal = RealVectorState((xg, yg))

        steer = SimpleDeltaSteering()
        sampler = Cartesian2DSampler(0, map_width, 0, map_height)
        collision_checker = Cartesian2DCollisionChecker(scene_map)
        rrt = RRT(x_init, x_goal, goal_radius, int(steer_delta), steer, sampler, collision_checker, num_nodes, max_planning_time)

        path, path_cost, stop_reason = _run_to_completion(rrt, max_planning_time)
        edges = rrt.tree_builder_.get_edges_in_order()
        path_found = len(path) > 0

        return {
            "edges": [[list(e[0]), list(e[1])] for e in edges],
            "path": [list(p) for p in path],
            "map_width": map_width,
            "map_height": map_height,
            "x_init": list(x_init.get_value()),
            "x_goal": list(x_goal.get_value()),
            "goal_radius": goal_radius,
            "path_cost": path_cost if path_found else None,
            "node_count": len(edges) + 1,
            "path_found": path_found,
            "stop_reason": stop_reason,
            "map_name": map_name,
        }
    finally:
        PLANNING_SEMAPHORE.release()


@app.get("/plan-rrtstar")
def stream_rrtstar(
    request: Request,
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
    max_planning_time: float | None = Query(None, ge=0.1, le=300),
):
    """Run RRT* step-by-step and stream full tree snapshots as Server-Sent Events."""
    _resolve_map_path(map_name)
    _acquire_planning_slot()
    is_disconnected = _disconnect_checker(request)

    def generate():
        try:
            try:
                scene_map, map_width, map_height = _load_scene_map(map_name)
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
                return

            x_init = RealVectorState((x0, y0))
            x_goal = RealVectorState((xg, yg))
            steer = SimpleDeltaSteering()
            sampler = Cartesian2DSampler(0, map_width, 0, map_height)
            collision_checker = Cartesian2DCollisionChecker(scene_map)

            rrtstar = RRTStar(
                x_init, x_goal, goal_radius, int(steer_delta), steer, sampler,
                eta, gamma_rrt,
                collision_checker, num_nodes,
                max_planning_time,
            )

            snapshot_extra = {
                "map_width": int(map_width),
                "map_height": int(map_height),
                "x_init": list(x_init.get_value()),
                "x_goal": list(x_goal.get_value()),
                "goal_radius": goal_radius,
                "map_name": map_name,
            }

            yield from _stream_rrtstar_events(
                rrtstar, batch_size, max_planning_time, snapshot_extra,
                path_builder=lambda path_list: [list(p) for p in path_list],
                is_disconnected=is_disconnected,
            )
        finally:
            PLANNING_SEMAPHORE.release()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/plan-differential-drive")
def compute_plan_differential_drive(
    map_name: str = "smile.png",
    goal_radius: int = Query(20, ge=1, le=500),
    num_nodes: int = Query(20000, ge=100, le=200000),
    x0: float = Query(30, ge=0),
    y0: float = Query(30, ge=0),
    xg: float = Query(30, ge=0),
    yg: float = Query(460, ge=0),
    max_planning_time: float | None = Query(None, ge=0.1, le=300),
    robot_radius: float = Query(8.0, ge=0.5, le=100),
    sampling_time: float = Query(20.0, ge=0.1, le=200),
):
    """Run the RRT planner for a DifferentialDriveRobot and return edges/path as
    [x, y, theta] states, mirroring /plan, with the path converted to chronological
    order (see _to_chronological_path) so the frontend can animate the robot along
    it directly, start to goal."""
    _resolve_map_path(map_name)
    _acquire_planning_slot()
    try:
        try:
            scene_map, map_width, map_height = _load_scene_map(map_name)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Failed to load map '{map_name}'.")

        x_init = RealVectorState((x0, y0, 0.0))
        x_goal = RealVectorState((xg, yg, 0.0))

        # wheel_radius/distance_wheels cancel out in DifferentialDriveRobot's unicycle
        # model (see main_differential_drive.py), so any nonzero values behave the same.
        wheel_radius = 1.0
        distance_wheels = 1.0
        steer_delta = sampling_time

        pose_sampler = DifferentialDrivePoseSampler(0, map_width, 0, map_height)
        steer = DifferentialDriveSteering(wheel_radius, distance_wheels, sampling_time)
        collision_checker = DifferentialDriveCollisionChecker(scene_map, robot_radius)
        rrt = RRT(x_init, x_goal, goal_radius, steer_delta, steer, pose_sampler,
                  collision_checker, num_nodes, max_planning_time)

        path, path_cost, stop_reason = _run_to_completion(rrt, max_planning_time)
        edges = rrt.tree_builder_.get_edges_in_order()
        path_found = len(path) > 0
        chronological_path = _to_chronological_path(path, x_init) if path_found else []

        return {
            "edges": [[list(e[0]), list(e[1])] for e in edges],
            "path": chronological_path,
            "map_width": map_width,
            "map_height": map_height,
            "x_init": list(x_init.get_value()),
            "x_goal": list(x_goal.get_value()),
            "goal_radius": goal_radius,
            "robot_radius": robot_radius,
            "path_cost": path_cost if path_found else None,
            "node_count": len(edges) + 1,
            "path_found": path_found,
            "stop_reason": stop_reason,
            "map_name": map_name,
        }
    finally:
        PLANNING_SEMAPHORE.release()


@app.get("/plan-differential-drive-rrtstar")
def stream_differential_drive_rrtstar(
    request: Request,
    map_name: str = "smile.png",
    goal_radius: int = Query(20, ge=1, le=500),
    num_nodes: int = Query(20000, ge=100, le=200000),
    x0: float = Query(30, ge=0),
    y0: float = Query(30, ge=0),
    xg: float = Query(30, ge=0),
    yg: float = Query(460, ge=0),
    batch_size: int = Query(100, ge=1, le=5000),
    gamma_rrt: float = Query(1000.0, ge=1.0),
    eta: float = Query(20.0, ge=1.0),
    max_planning_time: float | None = Query(None, ge=0.1, le=300),
    robot_radius: float = Query(8.0, ge=0.5, le=100),
    sampling_time: float = Query(20.0, ge=0.1, le=200),
):
    """Run RRT* step-by-step for a DifferentialDriveRobot and stream full tree
    snapshots as Server-Sent Events, mirroring /plan-rrtstar but with [x, y, theta]
    states and a chronological path (see /plan-differential-drive) for the frontend
    to animate."""
    _resolve_map_path(map_name)
    _acquire_planning_slot()
    is_disconnected = _disconnect_checker(request)

    def generate():
        try:
            try:
                scene_map, map_width, map_height = _load_scene_map(map_name)
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
                return

            x_init = RealVectorState((x0, y0, 0.0))
            x_goal = RealVectorState((xg, yg, 0.0))
            wheel_radius = 1.0
            distance_wheels = 1.0
            steer_delta = sampling_time

            pose_sampler = DifferentialDrivePoseSampler(0, map_width, 0, map_height)
            steer = DifferentialDriveSteering(wheel_radius, distance_wheels, sampling_time)
            collision_checker = DifferentialDriveCollisionChecker(scene_map, robot_radius)

            rrtstar = RRTStar(
                x_init, x_goal, goal_radius, steer_delta, steer, pose_sampler,
                eta, gamma_rrt,
                collision_checker, num_nodes,
                max_planning_time,
            )

            snapshot_extra = {
                "map_width": int(map_width),
                "map_height": int(map_height),
                "x_init": list(x_init.get_value()),
                "x_goal": list(x_goal.get_value()),
                "goal_radius": goal_radius,
                "robot_radius": robot_radius,
                "map_name": map_name,
            }

            yield from _stream_rrtstar_events(
                rrtstar, batch_size, max_planning_time, snapshot_extra,
                path_builder=lambda path_list: _to_chronological_path(path_list, x_init),
                is_disconnected=is_disconnected,
            )
        finally:
            PLANNING_SEMAPHORE.release()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serve the PixiJS frontend
app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static")
