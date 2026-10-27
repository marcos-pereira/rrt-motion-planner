#    This code is distributed WITHOUT ANY WARRANTY, without the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#   See the GNU Lesser General Public License for more details.

#   The license is distributed along with this repository or you can check
#   <http://www.gnu.org/licenses/> for more details.

# Contributors:
# marcos-pereira (https://github.com/marcos-pereira)

#!/usr/bin/env python
""" Benchmark the planning time of RRT and RRT* by running run_step() for a fixed number of
nodes, with the same planner setup used by webvis/server.py.

For each planner and seed, this prints the total time and the time per step at a few node
count checkpoints, the node count at which the goal radius was first reached, the best path
cost found, and a fingerprint of the final tree. With a fixed seed, the fingerprint and path
cost let a "before vs after" comparison of a speed-up check that the planner still builds the
same tree (or, when nearest-neighbor ties are broken differently, a statistically equivalent one).

Example:
    python3 benchmark_planners.py --nodes 10000 --checkpoints 2000 5000 10000 --seeds 2026 1 2
    python3 benchmark_planners.py --planner rrtstar --nodes 3000 --profile
"""

import argparse
import contextlib
import cProfile
import hashlib
import io
import pstats
import tempfile
import time
from pathlib import Path

from random_config import random
from Map import load_map
from RRT import RRT
from RRTStar import RRTStar
from RealVectorState import RealVectorState
from SimpleDeltaSteering import SimpleDeltaSteering
from Cartesian2DSampler import Cartesian2DSampler
from Cartesian2DCollisionChecker import Cartesian2DCollisionChecker
from DifferentialDrivePoseSampler import DifferentialDrivePoseSampler
from DifferentialDriveSteering import DifferentialDriveSteering
from DifferentialDriveCollisionChecker import DifferentialDriveCollisionChecker


def parse_arguments():
    """ Return the parsed command line arguments of the benchmark. """
    parser = argparse.ArgumentParser(description="Benchmark the planning time of RRT and RRT*.")
    parser.add_argument("--map", default="smile.png", help="map PNG file in python-scripts/")
    parser.add_argument("--robot", choices=["point", "differential-drive"], default="point")
    parser.add_argument("--planner", choices=["rrt", "rrtstar", "both"], default="both")
    parser.add_argument("--nodes", type=int, default=5000, help="number of nodes to grow the tree to")
    parser.add_argument("--checkpoints", type=int, nargs="*", default=None,
                        help="node counts at which to report the elapsed time (defaults to --nodes)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026],
                        help="random seeds, one benchmark run per seed and planner")
    parser.add_argument("--profile", action="store_true", help="print a cProfile report per run")
    return parser.parse_args()


def load_scene_map(map_name):
    """ Load the map's occupancy grid. load_map() writes a no_background.png next to the map,
    so the map is copied to a temporary directory to keep python-scripts/ untouched. """
    scripts_dir = Path(__file__).resolve().parent
    temporary_dir = Path(tempfile.mkdtemp())
    (temporary_dir / map_name).write_bytes((scripts_dir / map_name).read_bytes())

    return load_map(map_name, test=True, maps_dir=temporary_dir)


def build_planner(planner_name, robot, scene_map, max_num_nodes):
    """ Return an RRT or RRTStar planner configured as in webvis/server.py, with its defaults. """
    map_height, map_width = scene_map.shape

    if robot == "point":
        state_init = RealVectorState((40, 40))
        state_goal = RealVectorState((700, 550))
        goal_radius = 10
        steer_delta = 15
        steer = SimpleDeltaSteering()
        sampler = Cartesian2DSampler(0, map_width, 0, map_height)
        collision_checker = Cartesian2DCollisionChecker(scene_map)
    else:
        state_init = RealVectorState((30.0, 30.0, 0.0))
        state_goal = RealVectorState((30.0, 460.0, 0.0))
        goal_radius = 20
        # In webvis/server.py, steer_delta equals the sampling time
        steer_delta = 20.0
        steer = DifferentialDriveSteering(1.0, 1.0, steer_delta)
        sampler = DifferentialDrivePoseSampler(0, map_width, 0, map_height)
        collision_checker = DifferentialDriveCollisionChecker(scene_map, 8.0)

    if planner_name == "rrt":
        return RRT(state_init, state_goal, goal_radius, steer_delta, steer, sampler,
                   collision_checker, max_num_nodes)

    gamma_rrt = 1000.0
    eta = 20.0
    return RRTStar(state_init, state_goal, goal_radius, steer_delta, steer, sampler,
                   eta, gamma_rrt, collision_checker, max_num_nodes)


def compute_tree_fingerprint(planner):
    """ Return a short hash of the final tree's (node, parent) pairs, so two runs with the same
    seed can be compared for building exactly the same tree. """
    parent_pairs = sorted(planner.node_to_parent_.items())
    return hashlib.sha1(repr(parent_pairs).encode()).hexdigest()[:12]


def run_benchmark(planner_name, robot, scene_map, max_num_nodes, checkpoints, seed, profile):
    """ Grow the tree of one planner to max_num_nodes, printing its timings and results. """
    # Reseed so that every planner run draws the same sequence of samples
    random.seed(seed)
    planner = build_planner(planner_name, robot, scene_map, max_num_nodes)

    profiler = cProfile.Profile() if profile else None
    checkpoint_timings = []
    first_goal_node_count = None
    best_path_cost = float("inf")

    start_time = time.perf_counter()
    last_time = start_time
    last_node_count = planner.node_count_

    # The planners print on every goal hit, which would dominate the benchmark output
    with contextlib.redirect_stdout(io.StringIO()):
        if profiler is not None:
            profiler.enable()

        while planner.node_count_ < max_num_nodes:
            path_found, _, state_new = planner.run_step()

            if path_found:
                if first_goal_node_count is None:
                    first_goal_node_count = planner.node_count_
                best_path_cost = min(best_path_cost, planner.node_to_cost_[state_new.get_value()])

            if planner.node_count_ in checkpoints:
                current_time = time.perf_counter()
                step_time = (current_time - last_time) / (planner.node_count_ - last_node_count)
                checkpoint_timings.append((planner.node_count_, current_time - start_time, step_time))
                last_time = current_time
                last_node_count = planner.node_count_

        if profiler is not None:
            profiler.disable()

    total_time = time.perf_counter() - start_time

    print(f"⏱️  {planner_name} | robot={robot} | seed={seed} | {max_num_nodes} nodes in {total_time:.2f} s")
    for node_count, elapsed_time, step_time in checkpoint_timings:
        print(f"    n={node_count:>7}: total {elapsed_time:7.2f} s, {step_time * 1000:6.3f} ms/step since previous checkpoint")
    print(f"    first goal at n={first_goal_node_count}, best path cost={best_path_cost:.3f}, "
          f"tree fingerprint={compute_tree_fingerprint(planner)}")

    if profiler is not None:
        report = io.StringIO()
        pstats.Stats(profiler, stream=report).sort_stats("cumulative").print_stats(15)
        print(report.getvalue())

    return total_time, first_goal_node_count, best_path_cost


def main():
    arguments = parse_arguments()
    checkpoints = set(arguments.checkpoints or []) | {arguments.nodes}
    planner_names = ["rrt", "rrtstar"] if arguments.planner == "both" else [arguments.planner]

    scene_map = load_scene_map(arguments.map)

    for planner_name in planner_names:
        results = [run_benchmark(planner_name, arguments.robot, scene_map, arguments.nodes,
                                 checkpoints, seed, arguments.profile)
                   for seed in arguments.seeds]

        if len(results) > 1:
            mean_time = sum(result[0] for result in results) / len(results)
            path_costs = [result[2] for result in results if result[2] != float("inf")]
            mean_cost = sum(path_costs) / len(path_costs) if path_costs else float("inf")
            print(f"📊 {planner_name} over {len(results)} seeds: mean time {mean_time:.2f} s, "
                  f"goal reached in {len(path_costs)}/{len(results)} runs, mean best path cost {mean_cost:.3f}")
        print()


if __name__ == "__main__":
    main()
