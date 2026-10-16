"""One-to-one wrist association; never force a duplicate hand onto the other side.

Image-space association is a heuristic, not a guarantee under complete
occlusion. Ambiguous/unmatched detections are missing observations.
"""
from itertools import product
import numpy as np


def associate_hands(wrists, labels, pose_frame=None, previous=None, max_distance=.20):
    wrists = np.asarray(wrists,dtype=float).reshape(-1,2)
    if len(wrists)>2 or len(labels)!=len(wrists):
        raise ValueError("expected at most two detections and one label per detection")
    if not np.isfinite(max_distance) or max_distance<=0:
        raise ValueError("max_distance must be positive")
    previous = np.full((2,2),np.nan) if previous is None else np.asarray(previous,dtype=float)
    anchors = np.full((2,2),np.nan)
    if pose_frame is not None:
        for side,index in enumerate((15,16)):  # anatomical left, right
            point = np.asarray(pose_frame[index],dtype=float)
            if np.isfinite(point[:2]).all() and np.isfinite(point[3]) and point[3]>=.3:
                anchors[side] = point[:2]
    cost = np.full((len(wrists),2),np.inf)
    for k,wrist in enumerate(wrists):
        if not np.isfinite(wrist).all(): continue
        fallback = 1 if labels[k]=="Left" else 0 if labels[k]=="Right" else -1
        for side in (0,1):
            if np.isfinite(anchors[side]).all():
                distance = float(np.linalg.norm(wrist-anchors[side]))
                if distance <= max_distance:
                    temporal = (float(np.linalg.norm(wrist-previous[side]))
                                if np.isfinite(previous[side]).all() else 0)
                    cost[k,side] = distance+.1*min(temporal,max_distance)
            elif np.isfinite(previous[side]).all():
                distance = float(np.linalg.norm(wrist-previous[side]))
                if distance <= max_distance: cost[k,side]=distance
            elif fallback==side:
                cost[k,side]=.5*max_distance
    best,best_cost = tuple([-1]*len(wrists)),float("inf")
    for assignment in product((-1,0,1),repeat=len(wrists)):
        used=[s for s in assignment if s>=0]
        if len(used)!=len(set(used)): continue
        score=sum(max_distance if s<0 else cost[k,s] for k,s in enumerate(assignment))
        if score<best_cost: best,best_cost=assignment,score
    return [None if s<0 else s for s in best]
