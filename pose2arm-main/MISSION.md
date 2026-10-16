# Mission: Pose2Arm collision-free motion retargeting

## Why
Turn recorded human arm and finger motion into smooth OpenArm dual-arm trajectories that preserve the gesture without visible interpenetration or long safety freezes.

## Success looks like
- Explain and choose between sampling-based and optimization-based local avoidance.
- Generate a collision-free trajectory with bounded velocity and acceleration.
- Measure tracking error, clearance, planning time, and safety-layer intervention.

## Constraints
- Reuse the existing MuJoCo collision model and Windows Python environment.
- Keep the current 18D pose representation and trained policy compatible during this phase.
- Safety takes priority over exact imitation.

## Out of scope
- Global scene perception and arbitrary external obstacles.
- Redesigning or retraining the 18D policy in this lesson.
