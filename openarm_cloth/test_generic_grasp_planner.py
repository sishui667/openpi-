"""Hidden simulator object geometry must not influence the visual query model."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import mujoco
import numpy as np
from generic_object_scene import build_object_model
from generic_grasp_planner import plan_observed_grasp,validate_query_motion
from teleop_towel_dataset import build_arm_specs


class QueryContractTests(unittest.TestCase):
    def test_shapes_inertia_and_frozen_source_preserved(self):
        xml=Path(__file__).resolve().parent/'simulation/v1/droid_cube_scene_perception.xml'
        before=hashlib.sha256(xml.read_bytes()).hexdigest()
        box=build_object_model('box',[.06,.04,.05],mass=.12)
        cyl=build_object_model('cylinder',[.04,.04,.07],mass=.12)
        np.testing.assert_allclose(box.geom('droid_object_geom').size,[.03,.02,.025])
        np.testing.assert_allclose(cyl.geom('droid_object_geom').size,[.02,.035,0])
        self.assertTrue((box.body('droid_object').inertia>0).all())
        self.assertTrue((cyl.body('droid_object').inertia>0).all())
        self.assertEqual(hashlib.sha256(xml.read_bytes()).hexdigest(),before)

    def test_hidden_pose_size_and_type_cannot_change_query(self):
        model=build_object_model();data=mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model,data,0)
        spec=build_arm_specs(model)['right'];seed=data.qpos[spec.qpos].copy()
        observation=SimpleNamespace(shape='box',size_xyz=[.06,.04,.05],
                                    center_world=np.array([.42,-.06,.425]),yaw_world_rad=.2)
        snapshots=[]
        def fake(qmodel,qdata,*args,**kwargs):
            self.assertIsNot(qmodel,model)
            address=int(qmodel.joint('droid_object_freejoint').qposadr[0])
            snapshots.append((qmodel.geom('droid_object_geom').size.copy(),qdata.qpos[address:address+7].copy(),
                              qdata.qpos[build_arm_specs(qmodel)['right'].qpos].copy()))
            return dict(direction=np.array([-np.sin(.2),np.cos(.2),0]))
        with patch('generic_grasp_planner.plan_face_grasp',fake):
            first=plan_observed_grasp(model,data,'right',observation,seed,seed)
            model.geom('droid_object_geom').size[:]=[.4,.3,.2]
            model.geom('droid_object_geom').type[0]=mujoco.mjtGeom.mjGEOM_CYLINDER
            address=int(model.joint('droid_object_freejoint').qposadr[0]);data.qpos[address:address+3]=[9,-8,7]
            second=plan_observed_grasp(model,data,'right',observation,seed,seed)
        for a,b in zip(snapshots[0],snapshots[1]): np.testing.assert_array_equal(a,b)
        self.assertEqual(first['query_geometry'],second['query_geometry'])
        np.testing.assert_array_equal(data.qpos[address:address+3],[9,-8,7])

    def test_overwide_targets_do_not_run_ik(self):
        model=build_object_model();data=mujoco.MjData(model)
        spec=build_arm_specs(model)['right'];seed=data.qpos[spec.qpos].copy()
        observation=SimpleNamespace(shape='box',size_xyz=[.30,.20,.05],center_world=np.array([.42,-.06,.425]),yaw_world_rad=0)
        with patch('generic_grasp_planner.plan_face_grasp') as legacy:
            with self.assertRaises(RuntimeError): plan_observed_grasp(model,data,'right',observation,seed,seed)
            legacy.assert_not_called()

    def test_only_fitting_box_face_is_given_to_ik(self):
        model=build_object_model();data=mujoco.MjData(model)
        spec=build_arm_specs(model)['right'];seed=data.qpos[spec.qpos].copy()
        observation=SimpleNamespace(shape='box',size_xyz=[.14,.04,.05],center_world=np.array([.42,-.06,.425]),yaw_world_rad=0)
        def fake(*args,**kwargs):
            self.assertEqual(len(kwargs['closing_axes']),1)
            np.testing.assert_allclose(kwargs['closing_axes'][0],[0,1,0])
            return dict(direction=np.array([0,1,0]))
        with patch('generic_grasp_planner.plan_face_grasp',fake):
            result=plan_observed_grasp(model,data,'right',observation,seed,seed)
        self.assertAlmostEqual(result['query_geometry']['selected_width_m'],.04)

    def test_cylinder_uses_four_radial_axes_not_square_face_semantics(self):
        model=build_object_model();data=mujoco.MjData(model)
        spec=build_arm_specs(model)['right'];seed=data.qpos[spec.qpos].copy()
        observation=SimpleNamespace(shape='cylinder',size_xyz=[.04,.04,.07],center_world=np.array([.42,-.06,.435]),yaw_world_rad=0)
        def fake(*args,**kwargs):
            self.assertEqual(len(kwargs['closing_axes']),4)
            return dict(direction=np.array([0,1,0]))
        with patch('generic_grasp_planner.plan_face_grasp',fake):
            result=plan_observed_grasp(model,data,'right',observation,seed,seed)
        self.assertAlmostEqual(result['query_geometry']['selected_width_m'],.04)

    def revised_plan(self):
        model=build_object_model();data=mujoco.MjData(model)
        spec=build_arm_specs(model)['right'];seed=data.qpos[spec.qpos].copy()
        observation=SimpleNamespace(shape='box',size_xyz=[.06,.04,.05],center_world=np.array([.8,.8,.425]),yaw_world_rad=0.)
        with patch('generic_grasp_planner.plan_face_grasp',return_value=dict(direction=np.array([0,1,0]))):
            plan=plan_observed_grasp(model,data,'right',observation,seed,seed)
        return model,data,plan,seed,observation

    def test_revised_path_uses_and_restores_private_robot_state(self):
        model,data,plan,seed,observation=self.revised_plan()
        execution=data.qpos.copy();private=plan['query_data'].qpos.copy()
        record=validate_query_motion(plan,seed,seed,observation,finger_position=.044)
        self.assertGreaterEqual(record['samples'],21)
        self.assertEqual(record['max_new_penetration_mm'],0.)
        np.testing.assert_array_equal(data.qpos,execution)
        np.testing.assert_array_equal(plan['query_data'].qpos,private)

    def test_revised_ik_outside_joint_limits_is_not_executed(self):
        model,data,plan,seed,observation=self.revised_plan()
        goal=seed.copy();goal[0]=plan['query_spec'].q_range[0,1]+.1
        with self.assertRaises(RuntimeError): validate_query_motion(plan,seed,goal,observation)

    def test_open_descent_cannot_excuse_penetrating_the_visual_target(self):
        model,data,plan,seed,observation=self.revised_plan()
        qdata=plan['query_data'];qm=plan['query_model']
        mujoco.mj_forward(qm,qdata)
        finger=plan['query_finger_geoms'][0]
        # Mesh origin is behind the physical finger surface; overlap the
        # calibrated pad itself, not empty space around the mesh origin.
        observation.center_world=(qdata.geom_xpos[finger]+
            qdata.geom_xmat[finger].reshape(3,3)@np.array([0,-.008,.045]))
        with self.assertRaises(RuntimeError): validate_query_motion(plan,seed,seed,observation,finger_position=.044)
        record=validate_query_motion(plan,seed,seed,observation,finger_position=.044,ignore_target_contacts=True)
        self.assertTrue(record['target_contacts_ignored'])


if __name__=='__main__': unittest.main()
