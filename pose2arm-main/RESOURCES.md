# Pose2Arm Motion Planning Resources

## Knowledge

- [OMPL RRT-Connect class reference](https://ompl.kavrakilab.org/core/classompl_1_1geometric_1_1RRTConnect.html)
  Official API reference for the bidirectional sampling-based planner. Use for: planner state, range, and integration concepts.
- [MoveIt CHOMP tutorial](https://moveit.picknik.ai/main/doc/how_to_guides/chomp_planner/chomp_planner_tutorial.html)
  Official explanation of trajectory initialization, smoothness weights, obstacle weights, clearance, and recovery settings.
- [TrajOpt paper](https://rll.berkeley.edu/~sachin/papers/Schulman-IJRR2014.pdf)
  Primary paper on sequential convex trajectory optimization and continuous-time collision constraints.
- [MoveIt motion-planning concepts](https://moveit.picknik.ai/main/doc/concepts/motion_planning.html)
  Official overview of collision checking, planner pipelines, and time parameterization.

## Wisdom (Communities)

- [MoveIt GitHub discussions](https://github.com/moveit/moveit2/discussions)
  Maintainer and practitioner feedback for planner configuration and collision-scene integration.

## Gaps

- A published benchmark matching this exact OpenArm bimanual imitation trajectory and MuJoCo model.
