"""Colour-independent tabletop box/cylinder perception and physical grasp.

Fixtures set the world; the controller receives RGB-D and robot encoders only.
No learned semantic recognition and no claim of arbitrary-object grasping.
The original red-cube demo and frozen scene files remain unchanged.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time
import cv2
import glfw
import mujoco
from mujoco import viewer
import numpy as np

from calibration_simulation import render_rgb_depth,body_transform
from demo_grasp_viewer import (initialize_observation_posture,load_chest_calibration,
                               repair_left_finger_position_servos,_NullViewer,
                               PAD_CONTACT_PROXY_LOCAL)
from generic_object_scene import build_object_model,PlacementFixture,TABLE_RIM_HEIGHT,PLACEMENT_INSET
from tabletop_fixture import FIXTURE_SIZES,sample_fixture
from generic_grasp_planner import plan_observed_grasp,validate_query_motion
from handeye_grasp_red_cube import (finger_joint_indices,latest_handeye_report,load_handeye,
                                    quat_from_body,solve_target_q,solve_gripper_contact_proxy_q)
from tabletop_perception import detect_tabletop_objects,select_tabletop_object,TargetMovedError
from task_workspace import TABLE_TOP,CONFIG
from teleop_towel_dataset import build_arm_specs,build_finger_actuators,apply_controller

ROOT=Path(__file__).resolve().parent
COLOURS=dict(red=(.85,.12,.08,1),blue=(.08,.25,.85,1),green=(.1,.8,.2,1),
             yellow=(.95,.72,.05,1),black=(.03,.03,.03,1),white=(.9,.9,.9,1),brown=(.3,.18,.08,1))
TABLE_BOUNDS=tuple((float(c-h+.002),float(c+h-.002))
                   for c,h in zip(CONFIG['table_center_m'][:2],CONFIG['table_half_size_m'][:2]))


def setup_fixture(args):
    """Simulation setup ONLY; no returned shape/size/pose goes to vision/IK."""
    if getattr(args,'random_fixture',False):
        if args.size is not None:
            raise ValueError('--random-fixture cannot be combined with a supplied --size')
        fixture=sample_fixture()
        fixture_shape=fixture.shape;size=np.asarray(fixture.size_xyz,dtype=float);colour=fixture.colour
    else:
        fixture_shape=args.shape;colour=args.color
        size=np.asarray(args.size if args.size is not None else FIXTURE_SIZES[fixture_shape],dtype=float)
    shape='cylinder' if fixture_shape=='cylinder' else 'box'
    if shape=='cylinder' and abs(size[0]-size[1])>1e-8:
        raise ValueError('A cylinder needs equal x/y diameters')
    model=build_object_model(shape,size,COLOURS[colour])
    data=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,data,0)
    specs=build_arm_specs(model);actuators=build_finger_actuators(model)
    repair_left_finger_position_servos(model,actuators)
    initialize_observation_posture(model,data,specs)
    address=int(model.joint('droid_object_freejoint').qposadr[0])
    x,y,degrees=args.fixture_pose;yaw=np.deg2rad(degrees)
    data.qpos[address:address+3]=[x,y,TABLE_TOP+size[2]/2+.001]
    data.qpos[address+3:address+7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
    data.qvel[:]=0;mujoco.mj_forward(model,data)
    return model,data,specs,actuators


def run(args):
    model,data,specs,actuators=setup_fixture(args)
    # Discard fixture arguments before entering the controller.
    check=args.check;perception_only=args.perception_only;target_index=args.target_index
    report_path=args.report
    automatic=check or getattr(args,'auto_start',False)
    args=None
    dt=float(model.opt.timestep);holds={s:data.qpos[specs[s].qpos].copy() for s in specs}
    report=dict(scope='upright isolated opaque boxes/cylinders on a calibrated table',
                recognition_mode='automatic geometry from RGB-D; no preselected target shape',
                table_boundary=dict(physical_rim_height_m=TABLE_RIM_HEIGHT,
                                    placement_footprint_inset_m=PLACEMENT_INSET,
                                    placement_clamp_during_grasp=False),
                controller_inputs=['RGB-D','camera calibration','support plane','robot encoders'],
                forbidden_inputs=['fixture pose/size/colour/shape','simulator object state','segmentation IDs'],
                training_performed=False,physical_grasp_attempted=False,observations=[])
    report['source_hashes']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                             ('demo_tabletop_objects.py','tabletop_perception.py','tabletop_fixture.py','generic_object_scene.py',
                              'generic_grasp_planner.py','visual_grasp_planner.py')}
    renderers={};windows=[];active_view=None;last_ui=0.;arm=None
    controls=dict(start=automatic,quit=False,reset=False)
    placement_fixture=None if automatic else PlacementFixture(model,data)
    def key(code):
        if code in (glfw.KEY_Q,glfw.KEY_ESCAPE,27,113): controls['quit']=True
        elif code==glfw.KEY_SPACE: controls['start']=True
        elif code in (glfw.KEY_R,114) and not controls['start']: controls['reset']=True
    @contextmanager
    def session():
        if check:
            with _NullViewer() as v: yield v
            return
        with viewer.launch_passive(model,data,key_callback=key) as v:
            try: yield v
            except Exception as error:
                report.update(verdict='FAIL',error=f'{type(error).__name__}: {error}')
                if report_path:
                    report_path.parent.mkdir(parents=True,exist_ok=True)
                    report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
                print('TEST STOPPED: '+str(error),flush=True)
                while v.is_running() and not controls['quit']:
                    v.set_texts((None,None,'TEST STOPPED - NOT A SUCCESS\n'+str(error)+'\nQ or close window to exit',''))
                    v.sync();code=cv2.waitKey(1)&255
                    if code!=255: key(code)
                    time.sleep(.03)
                raise
    pace=dict(wall=time.perf_counter(),sim=float(data.time),steps=0)

    def step(commands,finger=.044):
        apply_controller(model,data,specs,actuators,commands,{'left':finger,'right':finger})
        mujoco.mj_step(model,data)
        if not np.isfinite(data.qpos).all(): raise RuntimeError('Nonfinite robot dynamics')
        pace['steps']+=1
        if not check and pace['steps']%max(1,round(1/(30*dt)))==0:
            delay=pace['wall']+float(data.time)-pace['sim']-time.perf_counter()
            if delay>0: time.sleep(delay)

    def pump(label):
        nonlocal last_ui
        if check or time.perf_counter()-last_ui<1/30: return False
        if controls['quit'] or not active_view.is_running(): raise KeyboardInterrupt
        active_view.set_texts((None,None,label,''));active_view.sync()
        code=cv2.waitKey(1)&255
        if code!=255: key(code)
        time.sleep(.005);last_ui=time.perf_counter()
        return True

    def move(side,goal,seconds,finger=.044):
        pace.update(wall=time.perf_counter(),sim=float(data.time),steps=0)
        start=data.qpos[specs[side].qpos].copy();steps=max(1,round(seconds/dt))
        for i in range(steps):
            a=(i+1)/steps;cmd=dict(holds);cmd[side]=start+(goal-start)*a*a*(3-2*a)
            step(cmd,finger);pump('MOVING '+side.upper())
        holds[side]=goal.copy()
        for _ in range(round(.35/dt)): step(holds,finger);pump('HOLDING '+side.upper())
        return float(np.max(np.abs(data.qpos[specs[side].qpos]-goal)))

    chest_transform,chest_matrix,calibration=load_chest_calibration()
    report['chest_calibration']=str(calibration)

    def snapshot(stage):
        if not report_path: return
        camera=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(model,camera)
        camera.lookat[:]=[.43,0,.45];camera.distance=1.15;camera.azimuth=130;camera.elevation=-30
        renderer=mujoco.Renderer(model,480,640)
        try:
            renderer.update_scene(data,camera=camera)
            rgb=renderer.render()
            report_path.parent.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(report_path.with_suffix(f'.{stage}.png')),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        finally: renderer.close()

    def capture(side=None,reference=None,record=True):
        camera='chest_d435i_preview' if side is None else f'v22_{side}_d405_rgb'
        h,w=(480,640) if side is None else (240,320)
        if camera not in renderers: renderers[camera]=mujoco.Renderer(model,h,w)
        if side is None: transform,matrix=chest_transform,chest_matrix
        else:
            handeye,matrix=load_handeye(latest_handeye_report(side),side)
            transform=body_transform(data,specs[side].body_id)@handeye
        rgb,depth=render_rgb_depth(renderers[camera],data,camera)
        candidates=detect_tabletop_objects(rgb,depth,matrix,transform,TABLE_TOP,TABLE_BOUNDS)
        observation_record=dict(camera=camera,candidates=[o.as_dict() for o in candidates])
        if record: report['observations'].append(observation_record)
        # Save a diagnostic overlay, not an oracle segmentation image.
        image=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        for index,obs in enumerate(candidates):
            contours,_=cv2.findContours(obs.mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image,contours,-1,(0,255,0),1)
            x,y,_,_=obs.bbox
            cv2.putText(image,f'AUTO {index}: {obs.shape_label} '+str(np.round(obs.size_xyz*1000).astype(int))+' mm',
                        (x,max(15,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,0),1)
        if report_path and record:
            report_path.parent.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(report_path.with_suffix(f'.{len(report["observations"])}.png')),image)
        if not check:
            title='GENERIC RGB-D '+camera
            if title not in windows: cv2.namedWindow(title,cv2.WINDOW_NORMAL);windows.append(title)
            cv2.imshow(title,image);code=cv2.waitKey(1)&255
            if code!=255: key(code)
        return select_tabletop_object(candidates,reference,target_index if reference is None else None)

    try:
        with session() as active_view:
            if not check:
                active_view.cam.lookat[:]=[.43,0,.43];active_view.cam.distance=1.4
                active_view.cam.azimuth=130;active_view.cam.elevation=-30
            if not automatic:
                print('PLACEMENT READY | boundary active while editing; release mouse before SPACE | R: restore object | Q: exit',flush=True)
                preview_time=0.
                placement_fixture.constrain(data,placement_active=True)
                while not controls['start']:
                    if controls['reset']:
                        placement_fixture.reset(data,placement_active=True);controls['reset']=False
                        print('PLACEMENT RESET | object restored; camera preview will refresh',flush=True)
                    step(holds)
                    placement_fixture.constrain(data,placement_active=True)
                    if pump('PLACE INSIDE AMBER OUTLINE / CYAN RIM\nBoundary applies during placement only\nRelease mouse, then SPACE to start\nR: restore object | Q: exit'):
                        placement_fixture.constrain(data,placement_active=True)
                    if time.perf_counter()-preview_time>.4:
                        preview_time=time.perf_counter()
                        try: capture(record=False)
                        except RuntimeError: pass
                # A moved fixture must be observed anew; preview estimates
                # are deliberately discarded, never reused as grasp targets.
                with active_view.lock():
                    active_view.perturb.active=0
                    active_view.perturb.active2=0
                placement_fixture.finish(data)
                pace.update(wall=time.perf_counter(),sim=float(data.time),steps=0)
                placement_fixture=None
            for _ in range(round(1.2/dt)): step(holds);pump('SETTLING')
            snapshot('initial')
            coarse=capture();report['coarse']=coarse.as_dict()
            print('AUTO RECOGNITION',json.dumps(coarse.as_dict()),flush=True)
            pump('AUTO RECOGNIZED: '+coarse.shape_label.upper()+' | CAMERA MEASURED GEOMETRY')
            if perception_only:
                report.update(verdict='PERCEPTION_ONLY',physical_grasp_attempted=False)
                return report
            arm='left' if coarse.center_world[1]>.02 else 'right'
            spec=specs[arm];ready=holds[arm].copy()
            search=np.array([np.clip(coarse.center_world[0]-.19,.18,.38),
                             np.clip(coarse.center_world[1]*.65,-.13,.13),.60])
            q_view,*_=solve_target_q(model,data,spec,search,quat_from_body(model,data,spec.body_id),ready,
                                    target_rotation=body_transform(data,spec.body_id)[:3,:3])
            move(arm,q_view,1.5)
            try: observed=capture(arm,coarse);source='chest_and_wrist'
            except RuntimeError as error:
                observed=coarse;source='chest_fallback';report['wrist_fallback_reason']=str(error)
            move(arm,ready,1.5)
            report.update(target=observed.as_dict(),target_source=source,selected_arm=arm)
            print('VISION TARGET',json.dumps(observed.as_dict()),flush=True)
            def progress(metrics):
                print('GRASP CANDIDATE',metrics,flush=True);pump('PLANNING FROM CAMERA GEOMETRY')
            plan=plan_observed_grasp(model,data,arm,observed,holds[arm],ready,progress=progress)
            report.update(query_geometry=plan['query_geometry'],planning_metrics=plan['metrics'])
            metrics=plan['metrics']
            if metrics['midpoint_error_m']>.008 or metrics['perpendicular_error_m']>.008 or metrics['max_new_penetration_mm']>1.:
                raise RuntimeError('Camera target has no sufficiently aligned collision-free grasp candidate')
            # Oracle reads start ONLY after target, query geometry and plan freeze.
            ob=model.body('droid_object').id;og=model.geom('droid_object_geom').id
            reference_z=float(data.xpos[ob,2]);reference_center=data.xpos[ob].copy()
            report['perception_error_m_evaluation_only']=float(np.linalg.norm(observed.center_world-reference_center))
            fgi=tuple(model.geom(f'openarm_{arm}_{s}_finger_collision').id for s in ('right','left'))
            fq=finger_joint_indices(model,arm)
            move(arm,plan['q_transit'],1.5);move(arm,plan['q_approach'],1.2)
            # Target identity/position is rechecked from cameras after approach.
            revised=[];target_moves=[]
            for camera_side in (None,arm):
                try: revised.append(capture(camera_side,observed))
                except TargetMovedError as error: target_moves.append(str(error));print('REOBSERVATION',error,flush=True)
                except RuntimeError as error: print('REOBSERVATION',error,flush=True)
            if not revised:
                if target_moves: raise RuntimeError('TARGET MOVED: '+target_moves[0]+'; restart after placement is complete')
                raise RuntimeError('Both cameras lost the selected target before descent; no stale-coordinate grasp')
            if revised:
                nearest=min(revised,key=lambda o:np.linalg.norm(o.center_world-observed.center_world))
                observed=replace(observed,center_world=nearest.center_world.copy())
                qm,qd,qs=plan['query_model'],plan['query_data'],plan['query_spec']
                q_revised,residual,_,revised_metrics=solve_gripper_contact_proxy_q(qm,qd,qs,plan['query_finger_geoms'],
                    plan['query_finger_indices'],PAD_CONTACT_PROXY_LOCAL,observed.center_world,1,1,
                    plan['span'],.044,plan['q_grasp'],reference_rotation=plan['rotation'],
                    closing_direction=plan['direction'],multi_start=False)
                report['revised_grasp_ik']=dict(residual=residual,**revised_metrics)
                if revised_metrics['midpoint_error_m']>.008 or revised_metrics['perpendicular_error_m']>.008:
                    raise RuntimeError('Reobserved target cannot be aligned accurately; no stale-plan descent')
                report['revised_descent_path']=validate_query_motion(plan,data.qpos[spec.qpos],q_revised,
                                                                    observation=observed,finger_position=.044)
                plan['q_grasp']=q_revised
            report['execution_target_after_reobservation']=observed.as_dict()
            report['physical_grasp_attempted']=True
            move(arm,plan['q_grasp'],1.5)
            convergence=move(arm,plan['q_grasp'],1.2,0.)
            confirmed=(np.all(data.qpos[fq[0]]>.008) and
                       np.all(np.abs(data.actuator_force[actuators[arm]])>.8) and convergence<.03)
            report['closure_confirmed_by_encoders_and_effort']=bool(confirmed)
            snapshot('grip')
            if not confirmed: raise RuntimeError('Gripper encoders/effort do not confirm a grip; no fake lift')
            # Closed robot encoders determine pad span, not actual object width.
            points=[data.geom_xpos[g]+data.geom_xmat[g].reshape(3,3)@p for g,p in zip(fgi,PAD_CONTACT_PROXY_LOCAL)]
            span=float(np.linalg.norm(points[0]-points[1]));finger=float(np.mean(data.qpos[fq[0]]))
            q_lift,lift_residual,_,lift_metrics=solve_gripper_contact_proxy_q(plan['query_model'],plan['query_data'],plan['query_spec'],
                plan['query_finger_geoms'],plan['query_finger_indices'],PAD_CONTACT_PROXY_LOCAL,
                observed.center_world+[0,0,.08],1,1,span,finger,plan['q_grasp'],
                reference_rotation=plan['rotation'],closing_direction=plan['direction'],multi_start=False)
            report['lift_ik']=dict(residual=lift_residual,**lift_metrics)
            if lift_metrics['midpoint_error_m']>.008 or lift_metrics['perpendicular_error_m']>.008:
                raise RuntimeError('Lift IK cannot maintain the observed grip geometry')
            report['lift_path']=validate_query_motion(plan,data.qpos[spec.qpos],q_lift,
                observation=observed,finger_position=finger,ignore_target_contacts=True)
            move(arm,q_lift,2.,0.)
            continuous=best=0.
            for _ in range(round(.8/dt)):
                step(holds,0.);pump('LIFT / PHYSICAL EVALUATION')
                contacted=set()
                for c in data.contact:
                    if og in (c.geom1,c.geom2): contacted.update(set((int(c.geom1),int(c.geom2)))&set(fgi))
                dz=float(data.xpos[ob,2]-reference_z)
                good=dz>.03 and len(contacted)==2
                continuous=continuous+dt if good else 0.;best=max(best,continuous)
            report.update(lift_m_evaluation_only=float(data.xpos[ob,2]-reference_z),
                          both_finger_contact_lift_s_evaluation_only=best,verdict='PASS' if best>=.5 else 'FAIL')
            snapshot('lift')
            if not check:
                while active_view.is_running() and not controls['quit']: pump('RESULT '+report['verdict']+' | Q / close window to exit');time.sleep(.01)
    except KeyboardInterrupt:
        report.setdefault('verdict','CANCELLED')
    except Exception as error:
        report.update(verdict='FAIL',error=f'{type(error).__name__}: {error}')
    finally:
        for renderer in renderers.values(): renderer.close()
        for title in windows: cv2.destroyWindow(title)
        if report_path:
            report_path.parent.mkdir(parents=True,exist_ok=True)
            report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape',choices=('cube','box','cylinder'),default='cylinder',help='Fixture only, never a perception hint')
    parser.add_argument('--random-fixture',action='store_true',help='Create a random supported test solid; never supply its class to vision/IK')
    parser.add_argument('--color',choices=sorted(COLOURS),default='blue',help='Fixture only, not a detection threshold')
    parser.add_argument('--size',type=float,nargs=3,default=None,metavar=('DX','DY','DZ'))
    parser.add_argument('--fixture-pose',type=float,nargs=3,default=(.43,-.10,15.),metavar=('X','Y','YAW'))
    parser.add_argument('--check',action='store_true',help='Headless; otherwise opens MuJoCo and camera views')
    parser.add_argument('--auto-start',action='store_true',help='Skip manual placement confirmation; headless checks always auto-start')
    parser.add_argument('--perception-only',action='store_true',help='Do not move the arm or claim grasp success')
    parser.add_argument('--target-index',type=int,help='Select among visually enumerated objects; not a simulator ID')
    parser.add_argument('--report',type=Path,default=ROOT/'outputs'/'generic_objects'/'last_run.json')
    options=parser.parse_args()
    result=run(options)
    raise SystemExit(0 if result['verdict'] in ('PASS','PERCEPTION_ONLY') else 1)
