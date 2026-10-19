"""Asynchronous inter-arm replanning; no camera future or object state used.

The private query model has proximity margins; the physical model is untouched.
RRT finds joint routes, not guaranteed human-faithful motions. Sweeps are sampled,
and locally unreachable requests remain explicitly blocked rather than forced.
"""
import copy
import threading
import time
import numpy as np
import mujoco
from rrt_connect_planner import BimanualRRTConnect


SIDES=('left','right')


class InterarmGeometry:
    def __init__(self,model,specs,margin=.008):
        self.model=copy.copy(model);self.data=mujoco.MjData(self.model)
        self.qpos=np.concatenate([specs[s].qpos for s in SIDES])
        self.limits=np.vstack([specs[s].q_range for s in SIDES])
        self.margin=float(margin)
        self.sides={}
        for geom in range(model.ngeom):
            name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[geom])) or ''
            for side,ctype,affinity in (('left',1,2),('right',2,1)):
                if name.startswith(f'openarm_{side}_') and (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                    self.sides[geom]=side
                    self.model.geom_margin[geom]=.02
        self.names={g:(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or '') for g in range(model.ngeom)}

    def distance(self,q):
        self.data.qpos[self.qpos]=q
        mujoco.mj_kinematics(self.model,self.data);mujoco.mj_collision(self.model,self.data)
        minimum=.04
        for c in self.data.contact:
            a,b=int(c.geom1),int(c.geom2)
            if a not in self.sides and b not in self.sides: continue
            # Two fingers of the same empty gripper may close together.
            if self.sides.get(a)==self.sides.get(b) and 'finger' in self.names[a] and 'finger' in self.names[b]: continue
            minimum=min(minimum,float(c.dist))
        return minimum

    def is_safe(self,q,grip=None): return self.distance(q)>=self.margin

    def edge(self,a,b):
        count=max(1,int(np.ceil(np.max(np.abs(b-a))/.025)))
        return all(self.is_safe(a+(b-a)*(i/count)) for i in range(count+1))


class TimedRRT(BimanualRRTConnect):
    def __init__(self,geometry,seed=19):
        super().__init__(geometry,step_size=.25,edge_resolution=.025,max_iterations=1800,
                         corridor_sigma=.35,seed=seed)
        self.low=geometry.limits[:,0].copy();self.high=geometry.limits[:,1].copy()
        self.deadline=float('inf')

    def _state_safe(self,q,grips):
        if time.perf_counter()>self.deadline: raise TimeoutError('planning time budget')
        return super()._state_safe(q,grips)

    def _shortcut(self,path,grips,attempts=100,preserve_detour=False):
        try: return super()._shortcut(path,grips,attempts=8,preserve_detour=preserve_detour)
        except TimeoutError: return path

    def search(self,start,goal,seconds=1.):
        self.deadline=time.perf_counter()+seconds
        # The caller locks inactive dimensions from the live mode's rate caps.
        # Do not disable FULL wrist mode by hard-coding a TWIST-only mask here.
        try:
            path,stats=self.plan(start,goal,np.linspace(start,goal,12),np.zeros((2,2)))
            return path,dict(solved=stats.solved,nodes=stats.nodes,iterations=stats.iterations)
        except TimeoutError:
            return None,dict(solved=False,reason='budget_exhausted')


class RealtimeBimanualPlanner:
    def __init__(self,model,data,specs):
        self.geometry=InterarmGeometry(model,specs)
        self.worker_geometry=InterarmGeometry(model,specs)
        self.rrt=TimedRRT(self.worker_geometry)
        self.qpos=self.geometry.qpos;self.model=model;self.specs=specs
        self.lock=threading.Lock();self.wake=threading.Event();self.closed=False;self.thread=None
        self.generation=0;self.pending=None;self.result=None;self.busy=False
        self.path=[];self.route_goal=None;self.last_request=0.;self.request_goal=None
        self.last_delta=np.zeros(14);self.settled_goal=None;self.active_mask=None
        self.status=dict(mode='tracking',clearance_m=float('nan'),planning_ms=0.,route_id=0)

    def reset(self):
        with self.lock:
            self.generation+=1;self.pending=None;self.result=None
        self.path=[];self.route_goal=None;self.request_goal=None;self.settled_goal=None
        self.last_delta=np.zeros(14);self.active_mask=None

    def close(self):
        self.closed=True;self.wake.set()
        if self.thread is not None: self.thread.join(timeout=2.)

    def _request(self,start,goal,encoders,steps):
        now=time.perf_counter()
        with self.lock:
            changed=self.request_goal is None or np.max(np.abs(goal-self.request_goal))>.35
            if self.busy and not changed: return
            if now-self.last_request<.25: return
            self.last_request=now;self.request_goal=goal.copy();self.generation+=1
            self.pending=(self.generation,start.copy(),goal.copy(),encoders.copy(),np.asarray(steps).copy())
            if self.thread is None:
                self.thread=threading.Thread(target=self._worker,name='interarm-route-planner',daemon=True)
                self.thread.start()
        self.wake.set()

    def _worker(self):
        while not self.closed:
            self.wake.wait(.2);self.wake.clear()
            with self.lock:
                packet=self.pending;self.pending=None
                if packet is None: continue
                self.busy=True
            generation,start,goal,encoders,steps=packet;begin=time.perf_counter()
            try:
                self.worker_geometry.data.qpos[:]=encoders
                self.rrt.low=self.geometry.limits[:,0].copy();self.rrt.high=self.geometry.limits[:,1].copy()
                fixed=np.asarray(steps)<=0
                self.rrt.low[fixed]=start[fixed];self.rrt.high[fixed]=start[fixed]
                # Infeasible endpoints cannot be solved by merely routing to
                # them; search a bounded nearby reference rather than forcing.
                requested=goal.copy()
                if not self.worker_geometry.is_safe(goal):
                    options=[]
                    for radius in (.10,.20,.35):
                        for _ in range(12):
                            q=goal+self.rrt.rng.uniform(-radius,radius,14)
                            q=np.clip(q,self.geometry.limits[:,0],self.geometry.limits[:,1])
                            q[fixed]=start[fixed]
                            if self.worker_geometry.is_safe(q): options.append(q)
                        if options: break
                    goal=min(options,key=lambda q:np.linalg.norm(q-requested)) if options else goal
                path,info=self.rrt.search(start,goal,seconds=1.)
                output=dict(generation=generation,start=start,goal=requested,path=path,
                            adjusted=bool(np.max(np.abs(goal-requested))>1e-5),
                            search_ms=(time.perf_counter()-begin)*1000,info=info)
            except Exception as error:
                output=dict(generation=generation,start=start,goal=goal,path=None,
                            search_ms=(time.perf_counter()-begin)*1000,info={'reason':str(error)})
            with self.lock:
                self.busy=False
                if generation==self.generation: self.result=output
                if self.pending is not None: self.wake.set()

    def coordinate(self,previous,goals,data,steps,fingers=None,enabled=True):
        begin=time.perf_counter()
        mask=np.asarray(steps)>0
        if self.active_mask is not None and not np.array_equal(mask,self.active_mask): self.reset()
        self.active_mask=mask.copy()
        start=np.concatenate([previous[s] for s in SIDES]);goal=np.concatenate([goals[s] for s in SIDES])
        encoders=data.qpos.copy()
        if fingers is not None:
            for side in SIDES:
                for number in (1,2):
                    joint=self.model.joint(f'openarm_{side}_finger_joint{number}').id
                    adr=int(self.model.jnt_qposadr[joint]);encoders[adr]=max(encoders[adr],fingers[side])
        self.geometry.data.qpos[:]=encoders
        actual=data.qpos[self.qpos].copy();distance=self.geometry.distance(actual)
        direct=start+np.clip(goal-start,-steps,steps)
        forecast=start+np.clip(goal-start,-3*steps,3*steps)
        with self.lock:
            completed=self.result;self.result=None
        if completed is not None and completed['path'] is not None:
            if np.max(np.abs(completed['goal']-goal))<.35 and np.max(np.abs(completed['start']-start))<.15 and self.geometry.edge(start,completed['path'][0]):
                self.path=[q.copy() for q in completed['path'][1:]];self.route_goal=completed['goal']
                self.settled_goal=completed['goal'].copy() if completed['adjusted'] else None
                self.status['route_id']+=1
        if completed is not None:
            self.status['worker_search_ms']=completed['search_ms']
            self.status['plan_failure']=completed['info'].get('reason','')
        if not enabled:
            self.reset();result=start;mode='paused'
        elif distance<self.geometry.margin:
            # An already touching robot cannot pass a collision-free edge
            # test whose first point is the current contact. Permit only a
            # small, monotonically separating recovery, not motion deeper in.
            self.path=[];best=None
            for joint in np.flatnonzero(steps>0):
                for sign in (-1.,1.):
                    candidate=actual.copy();candidate[joint]+=sign*min(.025,steps[joint])
                    candidate=np.clip(candidate,self.geometry.limits[:,0],self.geometry.limits[:,1])
                    clearance=self.geometry.distance(candidate)
                    if clearance<=distance+.00001: continue
                    middle=self.geometry.distance((actual+candidate)/2)
                    if middle<distance-.0001: continue
                    score=clearance-distance-.002*np.linalg.norm(candidate-goal)
                    if best is None or score>best[0]: best=(score,candidate)
            result=best[1] if best is not None else actual.copy()
            result[np.asarray(steps)<=0]=start[np.asarray(steps)<=0]
            mode='separating_contact' if best is not None else 'contact_needs_reset'
        elif self.path and self.route_goal is not None and np.max(np.abs(goal-self.route_goal))>.35 and self.geometry.edge(start,forecast) and self.geometry.edge(actual,direct):
            self.reset();result=direct;mode='tracking'
        elif self.path:
            waypoint=self.path[0];candidate=start+np.clip(waypoint-start,-steps,steps)
            if self.geometry.edge(start,candidate) and self.geometry.edge(actual,candidate):
                result=candidate;mode='detour'
                if self.route_goal is not None and np.max(np.abs(goal-self.route_goal))>.35:
                    self._request(result,goal,encoders,steps)
            else:
                self.path=[];self._request(start,goal,encoders,steps);result=start;mode='replanning'
        elif self.geometry.edge(start,forecast) and self.geometry.edge(actual,direct):
            result=direct;mode='tracking'
        elif self.settled_goal is not None and np.max(np.abs(goal-self.settled_goal))<.10:
            result=start;mode='infeasible_reference'
        else:
            self._request(start,goal,encoders,steps)
            # Continue safe progress toward the reference while the worker
            # searches, rather than freezing both arms at first warning.
            result=start;mode='planning'
            for alpha in (1.,.5,.25):
                candidate=start+alpha*(direct-start)
                if self.geometry.edge(start,candidate) and self.geometry.edge(actual,candidate):
                    result=candidate;break
        # Soften route corners when the softened step also passes collision
        # checks. Never cut an unvalidated corner merely for visual smoothness.
        delta=result-start
        soft=start+np.clip(delta,self.last_delta-.04,self.last_delta+.04)
        soft=start+np.clip(soft-start,-steps,steps)
        smoothing=mode=='detour' or self.status['mode']=='detour'
        if enabled and smoothing and distance>=self.geometry.margin and np.max(np.abs(soft-result))>1e-7 and self.geometry.edge(start,soft) and self.geometry.edge(actual,soft): result=soft
        if self.path and np.max(np.abs(self.path[0]-result))<1e-5: self.path.pop(0)
        result=np.clip(result,self.geometry.limits[:,0]+1e-7,self.geometry.limits[:,1]-1e-7)
        self.last_delta=result-start
        self.status.update(mode=mode,clearance_m=float(distance),planning_ms=(time.perf_counter()-begin)*1000)
        return {s:result[7*i:7*(i+1)].copy() for i,s in enumerate(SIDES)}
