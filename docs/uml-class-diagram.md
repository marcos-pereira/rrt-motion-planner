# UML Class Diagram — RRT / RRT* Planners

This diagram documents the abstract/concrete (Strategy) design used by
[`RRTPlanner.py`](../python-scripts/RRTPlanner.py),
[`RRT.py`](../python-scripts/RRT.py) and
[`RRTStar.py`](../python-scripts/RRTStar.py), together with the abstract
collaborator classes they depend on (`State`, `Steer`, `Sampler`,
`CollisionChecker`) and the concrete implementations of those collaborators
found in the repository.

Mermaid was chosen over Graphviz/Pyreverse because it lets the relationships
be curated by hand (only class names and key abstract methods, per the
project's "less abstraction, more clarity" style) instead of auto-dumping
every method/attribute, and it renders natively in GitHub and most Markdown
previewers without extra tooling. A pre-rendered PNG is also kept at
[`uml-class-diagram.png`](uml-class-diagram.png) for viewers without Mermaid
support:

![UML class diagram](uml-class-diagram.png)

```mermaid
classDiagram
    direction TB

    %% ---- Core abstractions ----
    class State {
        <<abstract>>
        +get_value()
        +set_value(value)
    }
    class Steer {
        <<abstract>>
        +steer(node1, node2, delta)
    }
    class Sampler {
        <<abstract>>
        +get_sample() State
    }
    class CollisionChecker {
        <<abstract>>
        +collision(state) bool
    }
    class RRTPlanner {
        <<abstract>>
        #state_init_ : State
        #state_goal_ : State
        #steer_ : Steer
        #sampler_ : Sampler
        #collision_checker_ : CollisionChecker
        +plan_found()*
        +run()*
        +run_step()*
        +plan()*
    }

    %% ---- Concrete states ----
    class RealVectorState {
        +get_value() tuple
        +set_value(value)
        +dimension() int
    }
    State <|-- RealVectorState

    %% ---- Concrete steering strategies ----
    class SimpleDeltaSteering {
        +steer(node1, node2, delta)
    }
    class DifferentialDriveSteering {
        +steer(node1, node2, delta)
    }
    class DifferentialDriveRandomControlSteering {
        +steer(node1, node2, delta)
    }
    Steer <|-- SimpleDeltaSteering
    Steer <|-- DifferentialDriveSteering
    Steer <|-- DifferentialDriveRandomControlSteering

    %% ---- Concrete samplers ----
    class Cartesian2DSampler {
        +get_sample() RealVectorState
    }
    class DifferentialDriveSampler {
        +get_sample() RealVectorState
    }
    class DifferentialDrivePoseSampler {
        +get_sample() RealVectorState
    }
    Sampler <|-- Cartesian2DSampler
    Sampler <|-- DifferentialDriveSampler
    Sampler <|-- DifferentialDrivePoseSampler

    %% ---- Concrete collision checkers ----
    class Cartesian2DCollisionChecker {
        +collision(state) bool
    }
    class DifferentialDriveCollisionChecker {
        +collision(state) bool
    }
    CollisionChecker <|-- Cartesian2DCollisionChecker
    CollisionChecker <|-- DifferentialDriveCollisionChecker

    %% ---- Planners ----
    class RRT {
        +plan_found()
        +run()
        +run_step()
        +plan()
    }
    class RRTStar {
        -gamma_rrt_
        -nearest_neighbor_eta_
        +plan_found()
        +run()
        +run_step()
        +plan()
        +rewire_tree(new_node, neighbors)
    }
    RRTPlanner <|-- RRT
    RRTPlanner <|-- RRTStar

    %% ---- Tree bookkeeping ----
    class TreeNode {
        -state_ : State
        -cost_ : float
        -parent_ : TreeNode
        -children_ : List~TreeNode~
        +get_state() State
        +get_parent() TreeNode
        +add_child(child)
    }
    class TreeBuilder {
        -adjacency_list_
        +add_node(parent_node, new_node)
    }

    %% ---- Composition / usage relationships ----
    RRTPlanner o-- "2" State : state_init_, state_goal_
    RRTPlanner o-- Steer : steer_
    RRTPlanner o-- Sampler : sampler_
    RRTPlanner o-- CollisionChecker : collision_checker_
    RRTPlanner *-- TreeBuilder : tree_builder_
    RRTPlanner "1" *-- "*" TreeNode : tree_nodes_
    TreeNode o-- State : state_
    TreeNode "1" --> "0..*" TreeNode : children_
    DifferentialDriveRandomControlSteering o-- Sampler : control_sampler_
```

## Notes on the design

- **`RRTPlanner`** is an `ABC` (template-method style base class) that
  implements the shared tree/graph bookkeeping (`nearest_node`, `path`,
  `cost_to_node`, node/edge storage, planning-time and node-budget checks)
  and declares four abstract hooks — `plan_found()`, `run()`, `run_step()`,
  `plan()` — that `RRT` and `RRTStar` must implement with their own
  expansion/rewiring logic.
- **`State`, `Steer`, `Sampler`, `CollisionChecker`** are Strategy-pattern
  interfaces injected into `RRTPlanner` via its constructor, so a planner is
  agnostic to the concrete configuration space (`RealVectorState`,
  differential-drive pose, etc.), steering method, sampling distribution, and
  collision representation.
- **`DifferentialDriveRandomControlSteering`** is itself a `Steer`
  implementation that internally composes a `Sampler` (to draw random control
  inputs), showing the same Strategy pieces being reused across layers.
- **`TreeNode`** holds a `State` plus parent/children pointers, forming the
  parent-pointer tree used to reconstruct paths and (in `RRTStar`) to
  propagate cost updates during rewiring. **`TreeBuilder`** is a separate,
  simpler adjacency-list structure kept in sync alongside it.
