# 让双臂绕开，而不是停住

当前安全投影只会沿原始直线退回。若人体目标要求双手互换左右位置，这条直线被另一只手臂堵住，算法只能停在边界。

## RRT-Connect

在 16 维状态空间（14 个臂关节 + 2 个夹爪开合量）从起点和终点各长一棵随机树。每次采样一个状态，向它延伸一小步；扩展过程持续做碰撞检查。两棵树连接时得到一条绕行路径。

## 轨迹优化

把一段轨迹的全部路点作为变量，最小化：

```text
跟踪误差 + 平滑代价 + 碰撞代价 + 限位代价
```

碰撞梯度把路点推离另一只机械臂，速度、加速度和 jerk 项让动作保持连贯。

> 推荐：先用滑动窗口轨迹优化；若窗口仍在碰撞或陷入局部最优，用 RRT-Connect 生成可行绕行初值，再交给优化器平滑。

## 快速自测

如果直线路径碰撞、但目标状态本身安全，哪种算法更擅长找到完全不同的绕行路线？答案是 RRT-Connect。

若已有一条可行但折线很多的路径，哪种算法更适合改善？答案是轨迹优化。

## 主要资料

- [OMPL RRT-Connect](https://ompl.kavrakilab.org/core/classompl_1_1geometric_1_1RRTConnect.html)
- [MoveIt CHOMP](https://moveit.picknik.ai/main/doc/how_to_guides/chomp_planner/chomp_planner_tutorial.html)
- [TrajOpt 论文](https://rll.berkeley.edu/~sachin/papers/Schulman-IJRR2014.pdf)

有任何一步不清楚，可以继续让 Codex 用当前轨迹中的具体帧解释。
