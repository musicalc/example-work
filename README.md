# Quintic Threefold Real-Line Search

This project is a research-oriented search framework for finding smooth quintic threefolds with unusually large numbers of real lines. It connects several distinct mathematical disciplines—including homotopy continuation, solution certification, and Riemannian optimization—into a single automated research pipeline.

## What I Accomplished

Rather than relying on isolated mathematical computations, this framework explores large regions of the moduli space autonomously. Key technical achievements include:

* **Geometric Parameterization:** Represents quintic threefolds as points on the unit sphere within a 126-dimensional parameter space.
* **Manifold Optimization:** Implements a Riemannian random walk with momentum, utilizing tangent-space projections, exponential maps, and parallel transport to navigate the space.
* **Homotopy Continuation:** Uses numerical algebraic geometry to track all 2,875 complex lines from one threefold to nearby candidates, avoiding the cost of solving systems from scratch.
* **Strict Certification & Topology:** Certifies every tracked solution before accepting a step and enforces topological parity constraints (the number of real lines must be odd) to reject numerical artifacts.
* **Autonomous Infrastructure:** Built for long-running execution with checkpointing, restart logic, and automated numerical-failure recovery mechanisms.

## Why This is Difficult

A generic smooth quintic threefold contains exactly 2,875 complex lines. Determining how many of these lines are real requires solving and certifying a large polynomial system. This project automates that heavy computational process, maintaining absolute correctness while continuously hill-climbing a 126-dimensional space to discover record-setting examples.

## How to Run

This script is written in Julia and relies heavily on `HomotopyContinuation.jl`. 

To start a new search, simply run the script. The program will automatically generate a fresh, certified seed to begin the search:

```
julia refactored_parameter_homotopy.jl
```

To resume a search from an existing state, pass a checkpoint file (`.jls`) as an argument:

```
julia refactored_parameter_homotopy.jl quintic_search_seed_example.jls
```

**Output:** Because the system is designed to run multiple parallel workers simultaneously, each instance writes its discoveries to an isolated CSV file (e.g., `master_lines_registry_<worker_id>.csv`) to prevent concurrent write collisions. Every accepted point logs the confirmed number of real lines alongside the corresponding 126 geometric parameters.
