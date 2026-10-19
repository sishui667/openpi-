"""Local latest-usable-test retention. No automatic model training or upload.

Only recognized, closed webcam test runs are recycled. Numerical summaries
survive so regression comparisons do not disappear along with large videos.
"""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime
import json
import os
from pathlib import Path
import re

import numpy as np

RUN_NAME = re.compile(r'^\d{8}_\d{6}_\d{6}$')
FAMILIES = ('live_human', 'live_bimanual')
SCHEMA = 'openarm_live_test_v1'


def _json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _write(path, payload):
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def _linked(path):
    info = path.lstat()
    return path.is_symlink() or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


def managed_run(run, outputs_root):
    run, outputs_root = Path(run).absolute(), Path(outputs_root).absolute()
    if run.parent.name not in FAMILIES or not RUN_NAME.fullmatch(run.name):
        raise ValueError('Not a timestamped live test run')
    if run.parent.parent != outputs_root:
        raise ValueError('Run is outside the designated outputs directory')
    for path in (outputs_root, run.parent, run):
        if not path.is_dir() or _linked(path):
            raise ValueError('Missing directory or linked/reparse path')
    if run.resolve().parent.parent != outputs_root.resolve():
        raise ValueError('Resolved run escapes outputs directory')
    # Never recycle a tree containing a junction or other reparse point.
    for base, directories, files in os.walk(run, followlinks=False):
        for name in directories + files:
            if _linked(Path(base) / name):
                raise ValueError('Linked/reparse entry in test directory')
    return run.resolve()


def _recognized_contents(run):
    allowed = {'dataset', 'summary.json', 'latency.csv', 'display.csv',
               'numeric_control.npz', 'STOP', 'optimization_report.json'}
    for entry in run.iterdir():
        if entry.name not in allowed:
            return False
        if entry.is_dir() != (entry.name == 'dataset'):
            return False
    dataset = run / 'dataset'
    if not dataset.is_dir():
        return False
    for entry in dataset.iterdir():
        if not entry.is_file():
            return False
        if entry.name not in ('manifest.json', 'video_timestamps.csv') and not re.fullmatch(
                r'(chunk_\d{5}\.npz|human_camera_\d{5}\.mp4)', entry.name):
            return False
    return True


def cleanup_candidates(current, outputs_root):
    current = managed_run(current, outputs_root)
    result = []
    for family in FAMILIES:
        parent = Path(outputs_root) / family
        if not parent.is_dir() or _linked(parent):
            continue
        for entry in parent.iterdir():
            if entry.name >= current.name:
                continue
            try:
                run = managed_run(entry, outputs_root)
                manifest = _json(run / 'dataset' / 'manifest.json')
                if manifest.get('schema') == SCHEMA and manifest.get('finished') is True and _recognized_contents(run):
                    result.append(run)
            except (ValueError, OSError, json.JSONDecodeError):
                continue
    return sorted(result, key=lambda path: path.name)


def analyze_useful(dataset):
    manifest = _json(dataset / 'manifest.json')
    bilateral = len(manifest.get('arms', [])) == 2
    useful = fresh_count = 0
    left_hands = right_hands = 0
    angle_errors = []
    wrist_events = {side: dict(rejected_observations=0, joint_limit_observations=0) for side in ('left','right')}
    has_wrist_diagnostics = False
    for file in sorted(dataset.glob('chunk_*.npz')):
        with np.load(file, allow_pickle=False) as values:
            fresh = values['fresh_source'] & ~values['paused'] & ~values['startup']
            fresh_count += int(fresh.sum())
            if bilateral:
                full = fresh & values['side_pose_valid'].all(axis=1) & values['side_hand_valid'].all(axis=1)
                left_hands += int((fresh & values['side_hand_valid'][:, 0]).sum())
                right_hands += int((fresh & values['side_hand_valid'][:, 1]).sum())
                angle_errors.append(values['side_angle_errors_deg'][full])
                for index,side in enumerate(('left','right')):
                    if f'{side}_wrist_rejected' in values:
                        has_wrist_diagnostics = True
                        mask=fresh & values['side_pose_valid'][:,index] & values['side_hand_valid'][:,index]
                        wrist_events[side]['rejected_observations'] += int(np.sum(mask & values[f'{side}_wrist_rejected']))
                        wrist_events[side]['joint_limit_observations'] += int(np.sum(mask & values[f'{side}_wrist_limited']))
            else:
                full = fresh & values['pose_valid'] & values['hand_valid']
            useful += int(full.sum())
    errors = np.concatenate(angle_errors) if angle_errors else np.array([])
    finite = errors[np.isfinite(errors)]
    return dict(eligible=useful >= 20, useful_rows=useful, fresh_rows=fresh_count,
                definition='fresh, unpaused pose AND hand observations; both sides required for bimanual',
                hand_valid_rows=dict(left=left_hands, right=right_hands) if bilateral else None,
                wrist_diagnostics=wrist_events if has_wrist_diagnostics else None,
                controller_target_angle_error_p95_deg=float(np.percentile(finite, 95)) if finite.size else None,
                scope='controller target tracking, NOT human pose ground truth or grasp success')


def recycle_directory(path):
    """Force Windows 8+ recycling; fail closed, with no permanent-delete fallback.

    IFileOperation's FOFX_RECYCLEONDELETE explicitly requests the Recycle Bin;
    the legacy FOF_ALLOWUNDO merely preserves undo information "if possible".
    See Microsoft's IFileOperation::SetOperationFlags documentation.
    """
    if os.name != 'nt':
        raise RuntimeError('Recycle Bin implementation is Windows-only')
    import sys
    import uuid

    version = sys.getwindowsversion()
    if (version.major, version.minor) < (6, 2):
        raise OSError('Forced recycling requires Windows 8 or later')
    target = Path(path).absolute()
    if not target.is_dir() or _linked(target):
        raise ValueError('Recycle target must be an existing non-linked directory')
    target = target.resolve(strict=True)
    if target == Path(target.anchor):
        raise ValueError('Refusing to recycle a drive root')

    class GUID(ctypes.Structure):
        _fields_ = [('data1', wintypes.DWORD), ('data2', wintypes.WORD),
                    ('data3', wintypes.WORD), ('data4', ctypes.c_ubyte * 8)]

    def guid(value):
        return GUID.from_buffer_copy(uuid.UUID(value).bytes_le)

    hresult = ctypes.c_long
    ole32 = ctypes.WinDLL('ole32', use_last_error=True)
    shell32 = ctypes.WinDLL('shell32', use_last_error=True)
    ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    ole32.CoInitializeEx.restype = hresult
    ole32.CoUninitialize.argtypes = []
    ole32.CoUninitialize.restype = None
    ole32.CoCreateInstance.argtypes = [ctypes.POINTER(GUID), ctypes.c_void_p,
                                     wintypes.DWORD, ctypes.POINTER(GUID),
                                     ctypes.POINTER(ctypes.c_void_p)]
    ole32.CoCreateInstance.restype = hresult
    shell32.SHCreateItemFromParsingName.argtypes = [ctypes.c_wchar_p,
                                                  ctypes.c_void_p,
                                                  ctypes.POINTER(GUID),
                                                  ctypes.POINTER(ctypes.c_void_p)]
    shell32.SHCreateItemFromParsingName.restype = hresult

    def checked(result, operation):
        if result < 0:
            raise OSError(f'{operation} failed (HRESULT 0x{result & 0xffffffff:08X}); original data retained where possible')

    def method(interface, index, return_type, *argument_types):
        # IUnknown occupies vtable entries 0..2. IFileOperation indices below
        # follow the Windows SDK declaration in shobjidl_core.h.
        vtable = ctypes.cast(interface, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(return_type, ctypes.c_void_p, *argument_types)(vtable[index])

    # IFileOperation requires STA. If this thread is already MTA, abort rather
    # than falling back to SHFileOperation or an irreversible filesystem API.
    checked(ole32.CoInitializeEx(None, 0x2), 'STA CoInitializeEx')
    operation = ctypes.c_void_p()
    item = ctypes.c_void_p()
    try:
        clsid = guid('3AD05575-8857-4850-9277-11B85BDB8E09')
        iid = guid('947AAB5F-0A5C-4C13-B4D6-4BF7836FC9F8')
        checked(ole32.CoCreateInstance(ctypes.byref(clsid), None, 0x1,
                                      ctypes.byref(iid), ctypes.byref(operation)),
                'Create IFileOperation')
        # RECYCLEONDELETE + ADDUNDORECORD + EARLYFAILURE + NOERRORUI +
        # NOCONFIRMATION + SILENT. Never use FOF_ALLOWUNDO as the safety rule.
        flags = 0x00080000 | 0x20000000 | 0x00100000 | 0x0400 | 0x0010 | 0x0004
        checked(method(operation, 5, hresult, wintypes.DWORD)(operation, flags),
                'Set forced-recycle flags')
        shell_item_iid = guid('43826D1E-E718-42EE-BC55-A1E261C37BFE')
        checked(shell32.SHCreateItemFromParsingName(str(target), None,
                                                    ctypes.byref(shell_item_iid),
                                                    ctypes.byref(item)),
                'Create source IShellItem')
        checked(method(operation, 18, hresult, ctypes.c_void_p, ctypes.c_void_p)(
            operation, item, None), 'Queue recycle operation')
        checked(method(operation, 21, hresult)(operation), 'Perform forced recycling')
        aborted = wintypes.BOOL()
        checked(method(operation, 22, hresult, ctypes.POINTER(wintypes.BOOL))(
            operation, ctypes.byref(aborted)), 'Check recycle cancellation')
        if aborted.value or target.exists():
            raise OSError(f'Recycling not confirmed; operation aborted or source still exists: {target}')
    finally:
        if item.value:
            method(item, 2, wintypes.ULONG)(item)
        if operation.value:
            method(operation, 2, wintypes.ULONG)(operation)
        ole32.CoUninitialize()


def finish_test_cycle(run, outputs_root=None, apply=True, recycle=None):
    outputs_root = Path(outputs_root) if outputs_root is not None else Path(__file__).resolve().parent / 'outputs'
    report = dict(training_performed=False, upload_performed=False, recycled=[], skipped=[],
                  scope='offline diagnostics and latest usable recording retention; not learned weights')
    try:
        run = managed_run(run, outputs_root)
        manifest = _json(run / 'dataset' / 'manifest.json')
        summary = _json(run / 'summary.json')
        if manifest.get('schema') != SCHEMA or manifest.get('finished') is not True:
            raise ValueError('Current recording is not finalized')
        if manifest.get('run_error') or summary.get('run_error') or summary.get('error'):
            raise ValueError('Current test exited with an error')
        # New code always stores run_error, including a clean None; old single
        # recordings without an exit status cannot replace a known good test.
        if 'run_error' not in manifest:
            raise ValueError('No explicit clean application exit status')
        from validate_live_test_dataset import validate
        quality = validate(run / 'dataset', check_video=True)
        useful = analyze_useful(run / 'dataset')
        report.update(run=str(run), quality=quality, observations=useful)
        if not quality['valid'] or not quality['recording_complete'] or not useful['eligible']:
            raise ValueError('New recording incomplete, inconsistent, or fewer than 20 useful observations; old tests retained')
        targets = cleanup_candidates(run, outputs_root)
        report['eligible'] = True
        report['planned_recycle'] = [str(target) for target in targets]
        report['recommendations'] = ['Inspect recorded human video and compare target vs actual motion before changing parameters.',
                                     'Do not train on the robot action as ground truth for an incorrectly recognized human gesture.']
        history = outputs_root / 'test_cycle_history'
        if apply:
            history.mkdir(exist_ok=True)
            for target in targets:
                try:
                    # Revalidate the exact directory immediately before mutation.
                    managed_run(target, outputs_root)
                    if target not in cleanup_candidates(run, outputs_root):
                        raise ValueError('Candidate changed during validation')
                    snapshot = dict(run=str(target), replacement=str(run),
                                    recycled_at=datetime.now().isoformat(),
                                    manifest=_json(target / 'dataset' / 'manifest.json'))
                    for name in ('summary.json', 'optimization_report.json'):
                        if (target / name).is_file():
                            snapshot[name] = _json(target / name)
                    archive = history / f'{target.parent.name}_{target.name}.json'
                    snapshot['recycled'] = False
                    _write(archive, snapshot)
                    (recycle or recycle_directory)(target)
                    snapshot['recycled'] = True
                    _write(archive, snapshot)
                    report['recycled'].append(str(target))
                except (OSError, ValueError) as error:
                    report['skipped'].append(dict(path=str(target), reason=str(error)))
            _write(run / 'optimization_report.json', report)
            _write(history / f'{run.parent.name}_{run.name}.json', dict(run=str(run), manifest=manifest, summary=summary, report=report))
    except Exception as error:
        # Diagnostics/retention failure must not crash simulation shutdown or
        # discard the prior dataset. Never retry via permanent deletion.
        report.update(eligible=False, error=f'{type(error).__name__}: {error}')
    print('TEST CYCLE ' + json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--outputs-root', type=Path)
    parser.add_argument('--apply', action='store_true', help='Recycle recognized older runs; default is read-only')
    args = parser.parse_args()
    result = finish_test_cycle(args.run, args.outputs_root, args.apply)
    raise SystemExit(0 if result['eligible'] else 1)
