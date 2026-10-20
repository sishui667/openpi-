"""Reproducible camera/physics checks; fixture truth is evaluation only."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from demo_tabletop_objects import run,ROOT
from task_workspace import TABLE_TOP

CASES=(
    ('blue_cube','cube','blue',(.0576,.0576,.0576),(.43,-.10,15),False),
    ('green_box','box','green',(.070,.040,.055),(.43,.10,30),False),
    ('black_cylinder','cylinder','black',(.045,.045,.075),(.43,-.10,0),False),
    ('brown_cube','cube','brown',(.0576,.0576,.0576),(.43,-.10,15),True),
    ('white_box','box','white',(.065,.035,.050),(.45,.10,-25),True),
    ('yellow_cylinder','cylinder','yellow',(.050,.050,.060),(.40,-.08,0),True),
)


def benchmark(folder):
    folder.mkdir(parents=True,exist_ok=True);results=[]
    for name,shape,colour,size,pose,perception_only in CASES:
        options=SimpleNamespace(shape=shape,color=colour,size=size,fixture_pose=pose,check=True,
                                perception_only=perception_only,target_index=None,report=folder/f'{name}.json')
        with (folder/f'{name}.log').open('w',encoding='utf-8') as log,redirect_stdout(log): report=run(options)
        measurement=report.get('target',report.get('coarse'))
        expected_shape='cylinder' if shape=='cylinder' else 'box'
        center_error=dimensions_error=None;geometry_ok=False
        if measurement:
            # These known fixture commands are only used AFTER controller run.
            truth=np.array([pose[0],pose[1],TABLE_TOP+size[2]/2])
            center_error=float(np.linalg.norm(np.asarray(measurement['center_world'])-truth))
            observed_size=np.asarray(measurement['size_xyz']);target_size=np.r_[sorted(size[:2],reverse=True),size[2]]
            dimensions_error=float(np.max(np.abs(observed_size-target_size)))
            geometry_ok=measurement['shape']==expected_shape and center_error<.005 and dimensions_error<.005
        passed=geometry_ok and report['verdict']==('PERCEPTION_ONLY' if perception_only else 'PASS')
        result=dict(case=name,passed=bool(passed),test='perception' if perception_only else 'physical_grasp',
                    geometry_ok=bool(geometry_ok),center_error_m=center_error,size_error_max_m=dimensions_error,
                    verdict=report['verdict'],lift_m=report.get('lift_m_evaluation_only'),
                    contact_lift_s=report.get('both_finger_contact_lift_s_evaluation_only'),
                    report=str(options.report))
        results.append(result);print(json.dumps(result),flush=True)
        summary=dict(scope='Six fixed simulated cases, not a population success rate or real-camera accuracy',
                     passed=sum(item['passed'] for item in results),cases=results)
        (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs'/'generic_objects'/datetime.now().strftime('benchmark_%Y%m%d_%H%M%S'))
    args=parser.parse_args();summary=benchmark(args.output)
    raise SystemExit(0 if summary['passed']==len(CASES) else 1)
