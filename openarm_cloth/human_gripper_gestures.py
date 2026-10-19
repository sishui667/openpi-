"""Human hand gestures only. No scene/object information enters this module."""
import numpy as np


def fist_features(hand):
    """Four-finger curl from PIP angle and tip-to-wrist reach, scale independent.

    Thumb position is deliberately irrelevant: fist != thumb/index pinch.
    Ambiguous/degenerate observations are missing rather than invented fingers.
    """
    if hand is None: return None
    h=np.asarray(hand,dtype=float)
    if h.shape!=(21,3) or not np.isfinite(h).all(): return None
    scores=[];angles=[]
    for mcp,pip,dip,tip in ((5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20)):
        a=h[mcp]-h[pip];b=h[dip]-h[pip]
        norm=np.linalg.norm(a)*np.linalg.norm(b)
        palm=np.linalg.norm(h[mcp]-h[0])
        if norm<1e-10 or palm<1e-5: return None
        angle=float(np.degrees(np.arccos(np.clip(np.dot(a,b)/norm,-1,1))))
        reach=np.linalg.norm(h[tip]-h[0])/palm
        bend=np.clip((165-angle)/75,0,1)
        shortened=np.clip((1.85-reach)/.65,0,1)
        scores.append(float(.7*bend+.3*shortened));angles.append(angle)
    scores=np.asarray(scores)
    folded=int(np.sum(scores>=.6));extended=int(np.sum(scores<=.25))
    gesture='FIST' if folded>=3 else 'OPEN PALM' if extended>=3 else 'PARTIAL: HOLD'
    return dict(scores=scores,angles_deg=np.asarray(angles),folded=folded,
                extended=extended,gesture=gesture)


class FistGripperMapper:
    """Two consistent NEW detections to switch; 0.6 s loss returns open.

    Calling again with the same camera sequence never accumulates evidence.
    """
    def __init__(self,confirm_frames=2,lost_seconds=.6):
        self.state=0;self.candidate=None;self.count=0;self.last_seq=None
        self.last_seen=None;self.confirm_frames=confirm_frames;self.lost_seconds=lost_seconds

    def update(self,features,sequence,now):
        if features is None:
            self.candidate=None;self.count=0
            if self.last_seen is None or now-self.last_seen>self.lost_seconds: self.state=0
            return self.state
        if sequence==self.last_seq: return self.state
        self.last_seq=sequence;self.last_seen=now
        wanted=1 if features['folded']>=3 else 0 if features['extended']>=3 else None
        if wanted is None:
            self.candidate=None;self.count=0
        elif wanted==self.candidate: self.count+=1
        else: self.candidate=wanted;self.count=1
        if self.count>=self.confirm_frames: self.state=wanted
        return self.state
