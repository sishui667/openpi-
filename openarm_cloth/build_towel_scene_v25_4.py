# -*- coding: utf-8 -*-
"""在已验证的 OpenArm V23 MJCF 上增量创建可变形毛巾场景（V25.4 独立桌面碰撞层版）。

用法（Windows，激活 openpi 环境）：
    cd D:\\openpi\\openarm_cloth
    python .\\build_towel_scene_v25_4.py
    python .\\build_towel_scene_v25_4.py --view

只读原始 V23；在同一 simulation/v1 目录创建全新的 V25.4 XML 和 JSON 报告，
保留原有网格相对路径、机器人结构、三相机、原有 keyframe；不更改夹爪/相机安装。
MuJoCo Python 未安装时会明确报错，不会生成冒充已编译的结果。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

SOURCE_NAME = 'folding_scene_V23_THREE_CAMERA_PREVIEW_final01.xml'
OUTPUT_STEM = 'folding_scene_V25_4_TOWEL_SUPPORT'
SUPPORT_NAME = 'towel_support_v25_4'
TOWEL_NAME = 'folding_towel_v25_4'
TABLE_NAME = 'folding_table'
EXPECTED_CAMERAS = ('v22_left_d405_rgb', 'v22_right_d405_rgb', 'chest_d435i_preview')

# 所有尺寸为米，质量为 kg。先验证物理稳定性，再用真实毛巾标定。
TOWEL_X = 0.34         # X: 折叠方向 / 近远方向
TOWEL_Y = 0.42         # Y: 左右方向
GRID_NX = 13          # X 向顶点数
GRID_NY = 17          # Y 向顶点数
TOWEL_MASS = 0.12
TOWEL_RADIUS = 0.003   # 接触半径近似，不等于真实织物厚度
TABLE_EDGE_MARGIN = 0.035
TOWEL_START_GAP = 0.012
SIM_TIMESTEP = 0.0005
VALIDATE_SECONDS = 1.5
SUPPORT_LIFT = 0.005

# Same bimanual preparation pose used by check_dual_pose.py.
READY_QPOS = (
    -0.5541, -0.6094, 0.7053, 1.1711, -0.0003, -0.2704, 0.0170, 0.0200, 0.0200,
     0.5605,  0.6083, -0.6952, 1.1659, -0.0006,  0.2542, -0.0186, 0.0200, 0.0200,
)
TOWEL_COLLISION_BIT = 4  # 与原 V23 的默认 bit1 隔离，先只和专用桌面支撑层碰撞


def fmt(nums):
    return ' '.join(f'{float(x):.12g}' for x in nums)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_names(mujoco, model, obj_type, n):
    return [mujoco.mj_id2name(model, obj_type, i) for i in range(n)]


def select_output(scene_dir):
    n = 1
    while True:
        path = scene_dir / f'{OUTPUT_STEM}_{n:02d}.xml'
        if not path.exists() and not path.with_suffix('.json').exists():
            return path
        n += 1


def table_from_model(mujoco, model, data):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_NAME)
    if gid < 0:
        raise RuntimeError(f'未找到桌面 geom: {TABLE_NAME!r}。请确认 V23 中真实的桌面碰撞 geom 名称。')
    if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
        raise RuntimeError('folding_table 不是 box geom；未擅自猜测桌面表面形状。')
    R = data.geom_xmat[gid].reshape(3, 3)
    if max(abs(float(R[i, j]) - float(i == j)) for i in range(3) for j in range(3)) > 1e-4:
        raise RuntimeError('桌面并非世界坐标轴对齐；此版只支持水平、与 X/Y 轴平行的 box 桌面。')
    center = [float(v) for v in data.geom_xpos[gid]]
    half = [float(v) for v in model.geom_size[gid]]
    if (int(model.geom_contype[gid]) & 1) == 0 and (int(model.geom_conaffinity[gid]) & 1) == 0:
        raise RuntimeError('桌面碰撞位屏蔽了默认接触类型 1；不能保证毛巾落在桌面。请先检查 folding_table 的接触设置。')
    if TOWEL_X > 2 * (half[0] - TABLE_EDGE_MARGIN) or TOWEL_Y > 2 * (half[1] - TABLE_EDGE_MARGIN):
        raise RuntimeError(f'毛巾大于桌面安全范围：桌面半尺寸 {half[:2]}，请减少毛巾大小。')
    return {'geom_id': gid, 'center': center, 'half_size': half, 'top_z': center[2] + half[2]}


def ensure_simulation_options(root):
    """为 flex 接触显式采用更稳定的求解设置；只修改新场景，不修改 V23 源文件。"""
    option = root.find('option')
    if option is None:
        option = ET.Element('option')
        root.insert(0, option)
    previous = {
        'timestep': option.get('timestep'),
        'iterations': option.get('iterations'),
        'solver': option.get('solver'),
        'integrator': option.get('integrator'),
        'tolerance': option.get('tolerance'),
    }
    option.set('timestep', fmt([SIM_TIMESTEP]))
    option.set('solver', 'CG')
    option.set('integrator', 'implicitfast')
    option.set('iterations', '400')
    option.set('tolerance', '1e-6')
    # flex + 静态桌面接触时先不使用 noslip 迭代，减少额外约束耦合。
    option.set('noslip_iterations', '0')
    size = root.find('size')
    if size is None:
        size = ET.Element('size')
        root.insert(0, size)
    if size.get('memory') is None:
        size.set('memory', '96M')
    return previous, {
        'timestep': option.get('timestep'),
        'iterations': option.get('iterations'),
        'solver': option.get('solver'),
        'integrator': option.get('integrator'),
        'tolerance': option.get('tolerance'),
        'noslip_iterations': option.get('noslip_iterations'),
        'memory': size.get('memory'),
    }


def add_towel_support(root, table):
    """在真实桌面顶面重合位置加一层不可见、独立碰撞 bit 的静态支撑盒。

    这样可以把“毛巾-桌面接触”从 V23 原有 collision/default/exclude 设置中隔离出来。
    该支撑层只服务毛巾，不改变机器人与真实 folding_table 的原有碰撞。
    """
    worldbody = root.find('worldbody')
    if worldbody is None:
        raise RuntimeError('源 XML 没有 worldbody。')
    if any(e.get('name') == SUPPORT_NAME for e in root.iter()):
        raise RuntimeError(f'源场景中已有 {SUPPORT_NAME}；拒绝重复添加。')
    cx, cy, _ = table['center']
    hx, hy, _ = table['half_size']
    # Use a deep hidden support volume.  A paper-thin box at the table surface
    # lets the flex contact solver accumulate penetration before it reacts.
    hz = 0.020
    # 盒子的上表面与 folding_table 的真实 top_z 完全重合。
    support = ET.SubElement(worldbody, 'geom', {
        'name': SUPPORT_NAME,
        'type': 'box',
        # Raise only the hidden towel support; the visible tabletop remains unchanged.
        'pos': fmt([cx, cy, table['top_z'] + SUPPORT_LIFT - hz]),
        'size': fmt([hx, hy, hz]),
        'rgba': '0 0 0 0',
        'contype': str(TOWEL_COLLISION_BIT),
        'conaffinity': str(TOWEL_COLLISION_BIT),
        'condim': '3',
        'friction': '0.9 0.005 0.0001',
        'solref': '0.006 1',
        'solimp': '0.98 0.995 0.0001',
        'group': '5',
    })
    return support


def add_cloth(root, table, markers=True):
    worldbody = root.find('worldbody')
    if worldbody is None:
        raise RuntimeError('源 XML 没有 worldbody。')
    if any(e.get('name') == TOWEL_NAME for e in root.iter()):
        raise RuntimeError(f'源场景中已有 {TOWEL_NAME}；拒绝重复添加。')

    # 左腕为 +Y，右腕为 -Y；靠近机械臂的一边是 -X。
    cx, cy, _ = table['center']
    xmin = cx - TOWEL_X / 2
    xmax = cx + TOWEL_X / 2
    ymin = cy - TOWEL_Y / 2
    ymax = cy + TOWEL_Y / 2
    z0 = table['top_z'] + TOWEL_START_GAP

    grid = ET.SubElement(worldbody, 'flexcomp', {
        'name': TOWEL_NAME, 'type': 'grid', 'dim': '2',
        'count': f'{GRID_NX} {GRID_NY} 1',
        'spacing': fmt([TOWEL_X / (GRID_NX - 1), TOWEL_Y / (GRID_NY - 1), 0.03]),
        'pos': fmt([cx, cy, z0]),
        'mass': fmt([TOWEL_MASS]), 'radius': fmt([TOWEL_RADIUS]),
        'rgba': '0.10 0.55 0.85 1',
    })
    ET.SubElement(grid, 'contact', {
        # V25.4 用专用 bit4，只与 towel_support_v25_4 发生接触。
        # 暂时不让毛巾碰机器人/真实桌面，先把柔性体稳定性单独验证清楚。
        'selfcollide': 'none', 'internal': 'false',
        'contype': str(TOWEL_COLLISION_BIT),
        'conaffinity': str(TOWEL_COLLISION_BIT),
        'condim': '3',
        'friction': '0.9 0.005 0.0001',
        'solref': '0.006 1', 'solimp': '0.98 0.995 0.0001',
    })
    # 这里改回 MuJoCo 官方 drape 示例同类思路：由 2D continuum 同时承担拉伸和弯曲，
    # 不再叠加 edge equality，减少“边约束 + 弯曲弹性 + 接触”三套约束互相打架。
    # Do not add an edge equality layer here; flex elasticity already supplies
    # the in-plane constraint, and the extra equality layer destabilizes contact.
    ET.SubElement(grid, 'elasticity', {
        'young': '20000', 'poisson': '0.2',
        'thickness': '0.001', 'elastic2d': 'both', 'damping': '0.03',
    })

    if markers:
        for side, y, rgba in (
            ('left', ymax, '0.95 0.20 0.15 0.8'),
            ('right', ymin, '0.95 0.75 0.10 0.8'),
        ):
            ET.SubElement(worldbody, 'site', {
                'name': f'towel_v25_4_{side}_INITIAL_target', 'type': 'sphere',
                'pos': fmt([xmin, y, z0 + 0.006]),
                'size': '0.008', 'rgba': rgba,
            })
    return {
        'dimensions_m': [TOWEL_X, TOWEL_Y], 'mass_kg': TOWEL_MASS,
        'vertices': GRID_NX * GRID_NY, 'grid': [GRID_NX, GRID_NY],
        'radius_m': TOWEL_RADIUS,
        'initial_z_m': z0, 'near_edge_x_m': xmin, 'far_edge_x_m': xmax,
        'fold_axis_x_m': cx,
        'initial_left_grasp_xyz': [xmin, ymax, z0],
        'initial_right_grasp_xyz': [xmin, ymin, z0],
        'nominal_left_place_xyz': [xmax, ymax, table['top_z'] + 0.01],
        'nominal_right_place_xyz': [xmax, ymin, table['top_z'] + 0.01],
        'collision_bit': TOWEL_COLLISION_BIT,
        'note': 'V25.4 isolates cloth-table collision on a hidden support box; self-collision and robot-cloth contact are intentionally disabled for this baseline.',
    }



def validate_initial_towel_bounds(mujoco, model, table):
    """读取编译后的 flex 顶点世界坐标，确认初始毛巾整张位于桌面安全范围内。"""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, TOWEL_NAME)
    if fid < 0:
        raise AssertionError(f'找不到 flex: {TOWEL_NAME}')
    adr = int(model.flex_vertadr[fid])
    num = int(model.flex_vertnum[fid])
    pts = data.flexvert_xpos[adr:adr + num]
    if num != GRID_NX * GRID_NY:
        raise AssertionError(f'毛巾顶点数异常：{num} != {GRID_NX * GRID_NY}')
    xmin0 = float(pts[:, 0].min()); xmax0 = float(pts[:, 0].max())
    ymin0 = float(pts[:, 1].min()); ymax0 = float(pts[:, 1].max())
    zmin0 = float(pts[:, 2].min()); zmax0 = float(pts[:, 2].max())
    cx, cy, _ = table['center']
    hx, hy, _ = table['half_size']
    eps = 1e-6
    if xmin0 < cx - hx + TABLE_EDGE_MARGIN - eps or xmax0 > cx + hx - TABLE_EDGE_MARGIN + eps:
        raise AssertionError(f'毛巾 X 初始范围越出桌面安全区: [{xmin0:.6f}, {xmax0:.6f}]')
    if ymin0 < cy - hy + TABLE_EDGE_MARGIN - eps or ymax0 > cy + hy - TABLE_EDGE_MARGIN + eps:
        raise AssertionError(f'毛巾 Y 初始范围越出桌面安全区: [{ymin0:.6f}, {ymax0:.6f}]')
    if zmin0 - TOWEL_RADIUS <= table['top_z'] + 0.002:
        raise AssertionError(
            f'毛巾初始离桌面太近：vertex z={zmin0:.6f}, radius={TOWEL_RADIUS:.6f}, '
            f'table top={table["top_z"]:.6f}'
        )
    return {
        'vertex_count': num,
        'x_range_m': [xmin0, xmax0],
        'y_range_m': [ymin0, ymax0],
        'z_range_m': [zmin0, zmax0],
        'surface_clearance_m': zmin0 - TOWEL_RADIUS - table['top_z'],
    }


def _contact_ids(contact):
    """兼容不同 MuJoCo Python 绑定的 contact 字段访问方式。"""
    try:
        geoms = tuple(int(x) for x in contact.geom)
    except Exception:
        geoms = (int(getattr(contact, 'geom1', -1)), int(getattr(contact, 'geom2', -1)))
    try:
        flexes = tuple(int(x) for x in contact.flex)
    except Exception:
        flexes = (-1, -1)
    return geoms, flexes


def set_ready_pose(mujoco, model, data):
    """Set the robot to the bimanual towel-folding pose used by the viewer check."""
    if model.nq < len(READY_QPOS):
        raise AssertionError(f'Ready pose needs {len(READY_QPOS)} qpos values, model has {model.nq}.')
    data.qpos[:len(READY_QPOS)] = READY_QPOS
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def add_ready_keyframe(root, flex_dof_count):
    """Add a ready keyframe so opening the generated scene does not use robot zero pose."""
    keyframe = root.find('keyframe')
    if keyframe is None:
        keyframe = ET.SubElement(root, 'keyframe')
    if any(key.get('name') == 'towel_ready' for key in keyframe.findall('key')):
        raise RuntimeError('Generated XML already contains the towel_ready keyframe.')
    qpos = list(READY_QPOS) + [0.0] * int(flex_dof_count)
    ET.SubElement(keyframe, 'key', {
        'name': 'towel_ready',
        'qpos': fmt(qpos),
    })


def dynamic_stability_check(mujoco, model, table, seconds=VALIDATE_SECONDS):
    """无 Viewer 预跑，并明确验证“毛巾确实和专用支撑层产生接触”。"""
    import numpy as np
    data = mujoco.MjData(model)
    set_ready_pose(mujoco, model, data)
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, TOWEL_NAME)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, SUPPORT_NAME)
    if fid < 0 or sid < 0:
        raise AssertionError(f'找不到 flex/support: flex={fid}, support={sid}')
    adr = int(model.flex_vertadr[fid]); num = int(model.flex_vertnum[fid])
    nstep = max(1, int(round(seconds / float(model.opt.timestep))))
    sample_every = max(1, nstep // 40)
    worst_z = 1e9
    max_z = -1e9
    max_speed = 0.0
    support_contact_samples = 0
    peak_support_contacts = 0
    first_support_contact_s = None
    min_contact_dist = 1e9

    # 让位置执行器保持其 qpos0，避免稳定性预跑时机器人因 ctrl=0 自己运动；
    # 这不改变最终 XML，仅影响本次 headless 验证数据。
    for a in range(model.nu):
        jid = int(model.actuator_trnid[a, 0])
        if 0 <= jid < model.njnt:
            adr_q = int(model.jnt_qposadr[jid])
            if 0 <= adr_q < data.qpos.size:
                lo, hi = model.actuator_ctrlrange[a]
                target = float(data.qpos[adr_q])
                if int(model.actuator_ctrllimited[a]):
                    target = min(max(target, float(lo)), float(hi))
                data.ctrl[a] = target

    for i in range(nstep):
        mujoco.mj_step(model, data)
        if i % sample_every == 0 or i == nstep - 1:
            pts = np.asarray(data.flexvert_xpos[adr:adr+num], dtype=float)
            if not np.isfinite(pts).all() or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise AssertionError(f'动态预跑在 t={data.time:.4f}s 出现 NaN/Inf')
            worst_z = min(worst_z, float(pts[:, 2].min()))
            max_z = max(max_z, float(pts[:, 2].max()))
            if data.qvel.size:
                max_speed = max(max_speed, float(np.max(np.abs(data.qvel))))

            n_support = 0
            for k in range(data.ncon):
                con = data.contact[k]
                geoms, flexes = _contact_ids(con)
                if sid in geoms and fid in flexes:
                    n_support += 1
                    min_contact_dist = min(min_contact_dist, float(con.dist))
            if n_support:
                support_contact_samples += 1
                peak_support_contacts = max(peak_support_contacts, n_support)
                if first_support_contact_s is None:
                    first_support_contact_s = float(data.time)

    pts = np.asarray(data.flexvert_xpos[adr:adr+num], dtype=float)
    xmin, xmax = float(pts[:,0].min()), float(pts[:,0].max())
    ymin, ymax = float(pts[:,1].min()), float(pts[:,1].max())
    zmin, zmax = float(pts[:,2].min()), float(pts[:,2].max())
    span_x, span_y = xmax-xmin, ymax-ymin
    cx, cy, _ = table['center']; hx, hy, _ = table['half_size']

    if support_contact_samples == 0:
        raise AssertionError(
            '2.0 s 预跑中毛巾从未与专用支撑层产生 flex-geom contact。'
            '这说明不是刚度参数问题，而是当前 MuJoCo 版本/碰撞设置没有建立预期接触。')
    if span_x < 0.75*TOWEL_X or span_y < 0.75*TOWEL_Y:
        raise AssertionError(
            f'毛巾动态预跑后明显塌缩/翻折: span=({span_x:.4f},{span_y:.4f}) m')
    # 对“顶点中心”留 2 mm 数值余量；正常情况下应落在 top_z + radius 附近。
    if zmin < table['top_z'] - 0.002:
        raise AssertionError(
            f'即使使用独立支撑层仍发生穿透: final zmin={zmin:.5f} m, '
            f'table_top={table["top_z"]:.5f} m, contacts_peak={peak_support_contacts}')
    if zmax > table['top_z'] + 0.080:
        raise AssertionError(f'毛巾异常弹起/发散: final zmax={zmax:.5f} m')
    if xmin < cx-hx-0.005 or xmax > cx+hx+0.005 or ymin < cy-hy-0.005 or ymax > cy+hy+0.005:
        raise AssertionError('毛巾动态预跑后越出桌面边界')

    return {
        'seconds': float(data.time),
        'final_x_range_m': [xmin, xmax], 'final_y_range_m': [ymin, ymax],
        'final_z_range_m': [zmin, zmax], 'final_span_xy_m': [span_x, span_y],
        'minimum_sampled_vertex_z_m': worst_z,
        'maximum_sampled_vertex_z_m': max_z,
        'maximum_abs_qvel_sampled': max_speed,
        'support_contact_sample_count': support_contact_samples,
        'peak_support_contacts': peak_support_contacts,
        'first_support_contact_s': first_support_contact_s,
        'minimum_support_contact_dist_m': None if min_contact_dist == 1e9 else min_contact_dist,
        'status': 'PASS',
    }

def extend_qpos_keys(root, old_nq, added_nq):
    """毛巾网格新增的 3 轴滑块 DOF 初值为 0；原 V23 keyframe 不能丢弃。"""
    for key in root.findall('keyframe/key'):
        for attr, old_len, delta in (('qpos', old_nq, added_nq),):
            value = key.get(attr)
            if value is None:
                continue
            arr = value.split()
            if len(arr) != old_len:
                raise RuntimeError(f"keyframe {key.get('name', '')!r} {attr} 长度 {len(arr)} != 源模型 {old_len}")
            key.set(attr, value + (' ' + ' '.join(['0'] * delta) if delta else ''))
        # keyframe 的 qvel 为可选属性，也要匹配 nv；这里所有新增自由度都是 1-DOF 滑块。
        if key.get('qvel') is not None:
            raise RuntimeError('源 keyframe 含 qvel：本版拒绝猜测原速度布局，请先检查该 keyframe。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parent,
                        help='OpenArm 项目根目录（默认脚本所在目录）')
    parser.add_argument('--source', default=SOURCE_NAME,
                        help='simulation/v1 下原始 XML 文件名')
    parser.add_argument('--markers', action='store_true', help='显示两颗初始抓取角调试标记（默认关闭）')
    parser.add_argument('--view', action='store_true', help='成功编译后启动 Viewer')
    args = parser.parse_args()
    try:
        import mujoco
    except ImportError as exc:
        parser.error(f'没有安装 mujoco；请在 (openpi) 环境执行：{exc}')

    scene_dir = args.project.resolve() / 'simulation' / 'v1'
    source = Path(args.source)
    source = (scene_dir / source) if not source.is_absolute() else source
    if not source.is_file():
        parser.error(f'未找到已验证的场景：{source}')
    if source.parent.resolve() != scene_dir.resolve():
        parser.error('为保证相对网格路径正确，源文件必须放在项目 simulation/v1 目录。')
    output = select_output(scene_dir)
    source_hash_before = sha256(source)
    baseline = mujoco.MjModel.from_xml_path(str(source.resolve()))
    data = mujoco.MjData(baseline)
    mujoco.mj_forward(baseline, data)
    table = table_from_model(mujoco, baseline, data)
    before_cams = object_names(mujoco, baseline, mujoco.mjtObj.mjOBJ_CAMERA, baseline.ncam)
    if any(name not in before_cams for name in EXPECTED_CAMERAS):
        raise RuntimeError(f'V23 三相机缺失：应有 {EXPECTED_CAMERAS}，实际 {before_cams}')
    before_act = object_names(mujoco, baseline, mujoco.mjtObj.mjOBJ_ACTUATOR, baseline.nu)
    before_joint = object_names(mujoco, baseline, mujoco.mjtObj.mjOBJ_JOINT, baseline.njnt)

    if baseline.nflex != 0:
        raise RuntimeError('源 V23 已包含可变形对象；为避免叠加两块布，拒绝自动继续。')
    baseline_geoms = object_names(mujoco, baseline, mujoco.mjtObj.mjOBJ_GEOM, baseline.ngeom)
    older_towel = [name for name in baseline_geoms if name and ('towel' in name.lower() or 'cloth' in name.lower())]
    if older_towel:
        raise RuntimeError(f'源 V23 已有疑似毛巾/布料 geom {older_towel}，请先确认，不能直接叠加。')
    # 复制为新 XML 时保留源文件中的注释；原始文件从不写入。
    tree = ET.parse(source, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    root = tree.getroot()
    prev_opts, now_opts = ensure_simulation_options(root)
    # Keep the support after the flexcomp in worldbody order, matching MuJoCo's
    # stable flat-cloth contact ordering used by the headless validation.
    towel_info = add_cloth(root, table, markers=args.markers)
    add_towel_support(root, table)
    # 先在临时树中不带旧 keyframe 编译，以获知新 nq / 网格是否能编译；
    # XML 中所有资产路径都相对源文件目录，临时 XML 与最终 XML 放在相同目录。
    keyframe = root.find('keyframe')
    if keyframe is not None:
        root.remove(keyframe)
    probe = output.with_suffix('.probe.xml')
    if probe.exists():
        raise FileExistsError(f'探测 XML 已存在；不会覆盖：{probe}')
    try:
        tree.write(probe, encoding='utf-8', xml_declaration=True)
        model_probe = mujoco.MjModel.from_xml_path(str(probe.resolve()))
        initial_bounds = validate_initial_towel_bounds(mujoco, model_probe, table)
        dynamic_check = dynamic_stability_check(mujoco, model_probe, table)
    finally:
        if probe.exists():
            probe.unlink()
    if keyframe is not None:
        root.append(keyframe)
    new_joint = object_names(mujoco, model_probe, mujoco.mjtObj.mjOBJ_JOINT, model_probe.njnt)
    new_act = object_names(mujoco, model_probe, mujoco.mjtObj.mjOBJ_ACTUATOR, model_probe.nu)
    new_cams = object_names(mujoco, model_probe, mujoco.mjtObj.mjOBJ_CAMERA, model_probe.ncam)
    if before_joint != new_joint[:baseline.njnt]:
        raise AssertionError('原始机器人关节顺序改变；不输出。')
    if before_act != new_act or before_cams != new_cams:
        raise AssertionError('原始执行器或相机改变；不输出。')
    if model_probe.nflex != baseline.nflex + 1:
        raise AssertionError('新增 flex 数量不是 1；不输出。')
    delta_q = model_probe.nq - baseline.nq
    delta_v = model_probe.nv - baseline.nv
    if delta_q != 3 * GRID_NX * GRID_NY or delta_q != delta_v:
        raise AssertionError(f'柔性网格 DOF 异常：新增 nq={delta_q}, nv={delta_v}；不输出。')
    extend_qpos_keys(root, baseline.nq, delta_q)
    add_ready_keyframe(root, delta_q)
    ET.indent(tree, space='  ')
    tree.write(output, encoding='utf-8', xml_declaration=True)
    try:
        final = mujoco.MjModel.from_xml_path(str(output.resolve()))
        if (final.nq, final.nv, final.nu, final.ncam, final.nflex) != (
            model_probe.nq, model_probe.nv, baseline.nu, baseline.ncam, model_probe.nflex
        ):
            raise AssertionError('最终编译后的模型维度与探测模型不一致')
        if sha256(source) != source_hash_before:
            raise AssertionError('源 V23 文件发生修改，请停止检查')
        report = {
            'source_xml': str(source), 'source_sha256': source_hash_before,
            'output_xml': str(output), 'mujoco_version': mujoco.__version__,
            'compilation': 'PASS', 'source_unmodified': True,
            'table': table, 'towel': towel_info, 'initial_towel_bounds': initial_bounds,
            'dynamic_stability_check': dynamic_check,
            'original': {'nq': baseline.nq, 'nv': baseline.nv, 'njnt': baseline.njnt,
                         'nu': baseline.nu, 'ncam': baseline.ncam, 'nflex': baseline.nflex},
            'new': {'nq': final.nq, 'nv': final.nv, 'njnt': final.njnt,
                    'nu': final.nu, 'ncam': final.ncam, 'nflex': final.nflex},
            'cameras_preserved': new_cams, 'actuators_preserved': new_act,
            'previous_options': prev_opts, 'new_options': now_opts,
            'validation_scope': 'MJCF compile + 1.5 s headless flat-towel stability from the towel_ready bimanual pose using an isolated hidden support collision layer + original robot joints/actuators/cameras/source hash. NOT verified: self-collision, robot-cloth grasp, folding, camera calibration.',
        }
        output.with_suffix('.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
        )
    except Exception:
        # 编译不通过时保留 XML 以便定位，但不写成功报告。
        print(f'[ERROR] 新场景未通过编译或校验，请检查：{output}', file=sys.stderr)
        raise
    print('[PASS] V25.4 towel-support model compiled:', output)
    print('[PASS] V23 source SHA256 unchanged:', source_hash_before)
    print(f'[TABLE] top Z={table["top_z"]:.5f} m')
    print(f'[TOWEL] {GRID_NX}x{GRID_NY} vertices; nq +{delta_q}; nflex={final.nflex}')
    print('[TOWEL INITIAL BOUNDS]', initial_bounds)
    print('[TOWEL DYNAMIC CHECK]', dynamic_check)
    print(f'[PRESERVED] actuators={final.nu}, cameras={final.ncam}')
    print('[REPORT]', output.with_suffix('.json'))
    print('[NOTE] 已通过专用支撑层平铺稳定性预跑；下一步再给真实夹爪加入同一碰撞 bit，启用抓取。')
    if args.view:
        subprocess.run([sys.executable, '-m', 'mujoco.viewer', f'--mjcf={output}'], check=True)


if __name__ == '__main__':
    main()
