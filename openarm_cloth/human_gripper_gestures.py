"""Human hand gestures only. No scene/object information enters this module."""
import numpy as np


def image_extension_features(hand_norm, image_size):
    """Strong visible extension evidence, never a 2D-only fist detector.

    Use full-image x/y pixels, not MediaPipe's estimated z or square-normalized
    coordinates. A bent/short projected finger is ambiguous, not proof of curl.
    Tiny, clipped, edge-on and malformed hands supply no opening override.
    """
    if hand_norm is None or image_size is None: return None
    h=np.asarray(hand_norm,dtype=float);size=np.asarray(image_size,dtype=float)
    if h.shape not in ((21,2),(21,3)) or size.shape!=(2,): return None
    if not np.isfinite(h[:,:2]).all() or not np.isfinite(size).all() or np.min(size)<=0: return None
    # Landmarks near the image boundary can be hallucinated beyond clipped
    # fingertips; reserve context instead of trusting a predicted in-frame tip.
    if np.any(h[:,:2]<.03) or np.any(h[:,:2]>.97): return None
    p=h[:,:2]*size
    palm=np.linalg.norm(p[9]-p[0]);width=np.linalg.norm(p[5]-p[17])
    if palm<12 or width/palm<.35: return None
    extended=[];pip_angles=[];dip_angles=[];reaches=[]
    def angle(a,b):
        length=np.linalg.norm(a)*np.linalg.norm(b)
        if length<1e-8: return float('nan')
        return float(np.degrees(np.arccos(np.clip(np.dot(a,b)/length,-1,1))))
    for mcp in (5,9,13,17):
        pip,dip,tip=mcp+1,mcp+2,mcp+3
        axis=p[mcp]-p[0];base=np.linalg.norm(axis)
        pa=angle(p[mcp]-p[pip],p[dip]-p[pip])
        da=angle(p[pip]-p[dip],p[tip]-p[dip])
        reach=np.linalg.norm(p[tip]-p[0])/max(base,1e-8)
        forward=float(np.dot(p[tip]-p[mcp],axis)/max(base*base,1e-8))
        segments=np.linalg.norm(np.diff(p[[mcp,pip,dip,tip]],axis=0),axis=1)
        ordered=all(np.dot(p[b]-p[a],axis)>0 for a,b in ((mcp,pip),(pip,dip),(dip,tip)))
        strong=(base>=10 and pa>=160 and da>=155 and reach>=1.65 and forward>=.6
                and np.min(segments)>=2 and ordered)
        extended.append(bool(strong));pip_angles.append(pa);dip_angles.append(da);reaches.append(reach)
    return dict(extended=np.asarray(extended),pip_angles_deg=np.asarray(pip_angles),
                dip_angles_deg=np.asarray(dip_angles),reach=np.asarray(reaches))


def fist_features(hand, hand_norm=None, image_size=None):
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
    scores=np.asarray(scores);world_scores=scores.copy()
    projected=image_extension_features(hand_norm,image_size)
    source='world'
    if projected is not None:
        visible=projected['extended'];count=int(visible.sum())
        # Three clearly extended fingers can release an ambiguous world pose.
        # Contradicting a strong 3D fist requires all four visibly extended.
        required=4 if np.sum(world_scores>=.6)>=3 else 3
        if count>=required and np.sum(world_scores<=.25)<3:
            scores=np.where(visible,np.minimum(scores,.15),scores);source='image_extension'
    folded=int(np.sum(scores>=.6));extended=int(np.sum(scores<=.25))
    gesture='FIST' if folded>=3 else 'OPEN PALM' if extended>=3 else 'PARTIAL: HOLD'
    return dict(scores=scores,angles_deg=np.asarray(angles),folded=folded,
                extended=extended,gesture=gesture,world_scores=world_scores,source=source,
                image_extended=np.zeros(4,dtype=bool) if projected is None else projected['extended'])


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


def gesture_diagnostics(features):
    """Numeric/string arrays for NPZ audit; no object arrays or inferred truth."""
    f=features or {}
    return dict(gesture=np.asarray(f.get('gesture','NO HAND')),
                gesture_source=np.asarray(f.get('source','missing')),
                world_curl_scores=np.asarray(f.get('world_scores',[float('nan')]*4)),
                image_extended=np.asarray(f.get('image_extended',[False]*4),dtype=bool))
