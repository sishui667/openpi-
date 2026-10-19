import sys,unittest,time
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parent),r'D:\openpi\openarm_cloth',r'D:\openpi\pose2arm-main']
import numpy as np
import mujoco
from live_motion_scene import build_motion_scene
from teleop_towel_dataset import build_arm_specs
from realtime_bimanual_planner import InterarmGeometry,TimedRRT,RealtimeBimanualPlanner
from member_b_actuators import ActuatorController


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.m,self.d=build_motion_scene();self.specs=build_arm_specs(self.m)

    def test_query_model_does_not_mutate_physics(self):
        masks=self.m.geom_contype.copy();margins=self.m.geom_margin.copy();encoders=self.d.qpos.copy()
        g=InterarmGeometry(self.m,self.specs);g.data.qpos[:]=encoders
        self.assertTrue(g.is_safe(encoders[g.qpos]))
        np.testing.assert_array_equal(self.m.geom_contype,masks)
        np.testing.assert_array_equal(self.m.geom_margin,margins)
        np.testing.assert_array_equal(self.d.qpos,encoders)

    def test_clear_motion_and_pause(self):
        planner=RealtimeBimanualPlanner(self.m,self.d,self.specs)
        start={s:self.d.qpos[self.specs[s].qpos].copy() for s in ('left','right')}
        goals={s:start[s].copy() for s in start};goals['left'][3]+=.04
        steps=np.tile([.125,.125,.125,.125,.22,0,0],2)
        try:
            result=planner.coordinate(start,goals,self.d,steps,enabled=False)
            np.testing.assert_allclose(result['left'],start['left'])
            result=planner.coordinate(start,goals,self.d,steps)
            self.assertGreater(result['left'][3],start['left'][3])
            self.assertIsNone(planner.thread)
        finally: planner.close()

    def test_async_route_executes_without_collision_and_returns_to_goal(self):
        cases=np.load(Path(__file__).parent/'live_avoidance_test_cases.npz')
        # This deterministic safe-endpoint case has an obstructed direct path.
        start=cases['q0'].copy();goal=cases['goals'][1].copy()
        planner=RealtimeBimanualPlanner(self.m,self.d,self.specs)
        self.d.qpos[planner.qpos]=start
        self.assertFalse(planner.geometry.edge(start,goal))
        steps=np.tile([.125,.125,.125,.125,.22,0,0],2)
        largest=0.;checks=[];modes=[];deadline=time.perf_counter()+8
        try:
            while time.perf_counter()<deadline:
                before=start.copy()
                targets=dict(left=start[:7],right=start[7:]);goals=dict(left=goal[:7],right=goal[7:])
                result=planner.coordinate(targets,goals,self.d,steps)
                start=np.r_[result['left'],result['right']]
                self.assertTrue(planner.geometry.edge(before,start))
                self.assertLessEqual(np.max(np.abs(start-before)-steps),1e-9)
                np.testing.assert_allclose(start[[5,6,12,13]],0,atol=1e-10)
                self.d.qpos[planner.qpos]=start
                modes.append(planner.status['mode']);checks.append(planner.status['planning_ms'])
                if np.max(np.abs(start-goal))<.01: break
                time.sleep(.02)
            print('ASYNC RESULT',len(modes),np.max(np.abs(start-goal)),set(modes),'control check p95',np.percentile(checks,95),flush=True)
            self.assertLess(np.max(np.abs(start-goal)),.01)
            self.assertIn('detour',modes)
            self.assertGreater(planner.status['route_id'],0)
        finally: planner.close()

    def test_planner_timeout_does_not_return_invalid_path(self):
        g=InterarmGeometry(self.m,self.specs);g.data.qpos[:]=self.d.qpos
        start=self.d.qpos[g.qpos];rrt=TimedRRT(g)
        path,info=rrt.search(start,start+.01,seconds=-1)
        self.assertIsNone(path)
        self.assertEqual(info['reason'],'budget_exhausted')

    def test_full_wrist_mode_is_not_silently_locked(self):
        planner=RealtimeBimanualPlanner(self.m,self.d,self.specs)
        start=self.d.qpos[planner.qpos].copy();goal=start.copy();goal[5]+=.05;goal[13]+=.05
        try:
            result=planner.coordinate(dict(left=start[:7],right=start[7:]),dict(left=goal[:7],right=goal[7:]),self.d,np.tile([.125]*4+[.22]*3,2))
            self.assertGreater(result['left'][5],.02)
            self.assertGreater(result['right'][6],.02)
        finally: planner.close()

    def test_already_touching_pose_can_separate_instead_of_permanent_hold(self):
        cases=np.load(Path(__file__).parent/'live_avoidance_test_cases.npz');safe=cases['q0'];goal=cases['goals'][1]
        planner=RealtimeBimanualPlanner(self.m,self.d,self.specs)
        planner.geometry.data.qpos[:]=self.d.qpos
        colliding=None
        for alpha in np.linspace(0,1,80):
            q=safe+alpha*(goal-safe)
            if planner.geometry.distance(q)<-.0002: colliding=q;break
        self.assertIsNotNone(colliding)
        self.d.qpos[planner.qpos]=colliding
        before=planner.geometry.distance(colliding)
        steps=np.tile([.125,.125,.125,.125,.22,0,0],2)
        try:
            result=planner.coordinate(dict(left=colliding[:7],right=colliding[7:]),dict(left=safe[:7],right=safe[7:]),self.d,steps)
            after=planner.geometry.distance(np.r_[result['left'],result['right']])
            print('CONTACT RECOVERY',before,after,planner.status['mode'],flush=True)
            self.assertGreater(after,before)
            self.assertEqual(planner.status['mode'],'separating_contact')
        finally: planner.close()

    def test_stale_route_is_not_executed_after_human_changes_goal(self):
        planner=RealtimeBimanualPlanner(self.m,self.d,self.specs)
        start=self.d.qpos[planner.qpos].copy();old=start.copy();old[3]+=.4
        goal=start.copy();goal[3]-=.1
        planner.result=dict(path=[start,old],start=start,goal=old,adjusted=False,search_ms=1.,info={})
        try:
            planner.coordinate(dict(left=start[:7],right=start[7:]),dict(left=goal[:7],right=goal[7:]),self.d,np.tile([.125]*4+[.22,0,0],2))
            self.assertEqual(planner.status['route_id'],0)
            self.assertNotEqual(planner.status['mode'],'detour')
        finally: planner.close()

    def test_physical_following_keeps_window_control_independent_of_search(self):
        cases=np.load(Path(__file__).parent/'live_avoidance_test_cases.npz');goal=cases['goals'][1]
        planner=RealtimeBimanualPlanner(self.m,self.d,self.specs);actuators=ActuatorController(self.m,self.d)
        command=self.d.qpos[planner.qpos].copy();steps=np.tile([.125,.125,.125,.125,.22,0,0],2)
        collisions=0;begin=time.perf_counter();frames=0
        try:
            for _ in range(180):
                result=planner.coordinate(dict(left=command[:7],right=command[7:]),dict(left=goal[:7],right=goal[7:]),self.d,steps)
                command=np.r_[result['left'],result['right']]
                actuators.set_targets(result,dict(left=.044,right=.044))
                for _ in range(100): actuators.apply();mujoco.mj_step(self.m,self.d)
                for contact in self.d.contact:
                    a=self.m.body(int(self.m.geom_bodyid[contact.geom1])).name
                    b=self.m.body(int(self.m.geom_bodyid[contact.geom2])).name
                    if ('openarm_left_' in a and 'openarm_right_' in b) or ('openarm_right_' in a and 'openarm_left_' in b): collisions+=1
                frames+=1;time.sleep(.005)
                if np.max(np.abs(self.d.qpos[planner.qpos]-goal))<.025: break
            print('PHYSICAL FOLLOW',frames,'seconds',time.perf_counter()-begin,'interarm_contacts',collisions,'goal_error_rad',np.max(np.abs(self.d.qpos[planner.qpos]-goal)),flush=True)
            self.assertEqual(collisions,0)
            self.assertLess(np.max(np.abs(self.d.qpos[planner.qpos]-goal)),.025)
        finally: planner.close()


if __name__=='__main__': unittest.main()
