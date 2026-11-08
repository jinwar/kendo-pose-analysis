"""Experimental, footage-specific shinai shaft estimation. No learned shinai model.

Sharp segments: LSD edges gated by wrists, direction and bamboo-like color.
Blurred swings: darkening against the first pre-strike frame, searched over rays
near the hands. Those are exposure-blur axes, NOT instantaneous tip measurements.
No temporal interpolation. All endpoints describe visible support, not proven tips.
"""
import csv
import json
from pathlib import Path
from collections import Counter
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/strike_analysis'


def sample(image,points):
    return cv2.remap(image,points[...,0].astype('float32'),points[...,1].astype('float32'),
                     cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)


def sharp_segments(frame,grip,side,lines):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    candidates=[]
    for line in lines:
        a,b=line[:2].astype(float),line[2:].astype(float)
        if a[1]<b[1]:a,b=b,a
        u=b-a;length=np.linalg.norm(u)
        if length<70:continue
        u/=length;v=np.array([-u[1],u[0]])
        angle=float(np.degrees(np.arctan2(u[1],u[0])))
        if not ((-110<angle<-5) if side=='attacker' else (-175<angle<-95)):continue
        distance=abs(np.dot(grip-a,v));near=np.dot(a-grip,u);far=np.dot(b-grip,u)
        if distance>65 or far<130 or far>700 or near>220 or near< -100:continue
        xy=a+np.linspace(.05,.95,80)[:,None,None]*(b-a)+np.arange(-5,6)[None,:,None]*v
        pixels=sample(hsv,xy)
        tan=float(((pixels[:,:,0]>8)&(pixels[:,:,0]<42)&(pixels[:,:,1]>45)&(pixels[:,:,2]>45)).mean())
        if tan<.25:continue
        quality=length-.8*distance-.3*max(0,near)+50*tan
        candidates.append(dict(a=a,b=b,u=u,angle=angle,tan=tan,quality=quality))
    if not candidates:return None
    best=max(candidates,key=lambda x:x['quality']);a=best['a'];u=best['u'];v=np.array([-u[1],u[0]])
    # Join collinear supported edge pieces across small crossings/tape gaps.
    ends=[0,np.dot(best['b']-a,u)]
    for q in candidates:
        if abs(q['angle']-best['angle'])<3 and abs(np.dot(q['a']-a,v))<15:
            ends.extend([np.dot(q['a']-a,u),np.dot(q['b']-a,u)])
    return {'segment_xy':[(a+min(ends)*u).tolist(),(a+max(ends)*u).tolist()],
            'method':'edge_supported','angle_degrees':best['angle'],
            'bamboo_color_fraction':best['tan'],'quality_score':best['quality'],
            'endpoint_meaning':'Visible shaft segment; endpoints are not verified handle/tip.'}


def blurred_axis(frame,reference,grip):
    gray=cv2.GaussianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32),(0,0),5)
    delta=reference-gray
    angles,offsets=np.meshgrid(np.deg2rad(np.arange(-95,-9,2)),np.arange(-80,85,10),indexing='ij')
    angles=angles.ravel();offsets=offsets.ravel()
    u=np.stack([np.cos(angles),np.sin(angles)],axis=-1);v=np.stack([-np.sin(angles),np.cos(angles)],axis=-1)
    distance=np.arange(130,431,8)
    xy=grip+offsets[:,None,None]*v[:,None,:]+distance[None,:,None]*u[:,None,:]
    valid=(xy[:,:,1]>10)&(xy[:,:,1]<frame.shape[0]-10)&(xy[:,:,0]>10)&(xy[:,:,0]<frame.shape[1]-10)
    contrast=np.clip(sample(delta,xy),-15,20)
    scores=(contrast*valid).sum(axis=1)/valid.sum(axis=1).clip(1)
    scores[valid.sum(axis=1)<18]=-100
    k=int(np.argmax(scores));quality=float(scores[k])
    if quality<3:return {'method':'unresolved','segment_xy':None,'quality_score':quality}
    base=grip+offsets[k]*v[k]
    # Determine supported visible extent on the estimated blur axis. This is
    # not a fixed weapon length and is not claimed to locate the physical tip.
    d=np.arange(60,651,4);xy=base+d[:,None]*u[k]
    good=(xy[:,0]>=0)&(xy[:,0]<frame.shape[1])&(xy[:,1]>=0)&(xy[:,1]<frame.shape[0])
    profile=sample(delta,xy[None,:,:])[0]
    profile=np.convolve(profile,np.ones(5)/5,mode='same')
    support=((profile>max(2,quality*.22))&good).astype(np.uint8)
    support=cv2.morphologyEx(support[None,:],cv2.MORPH_CLOSE,np.ones((1,7),np.uint8))[0]&good
    groups=np.split(np.flatnonzero(support),np.flatnonzero(np.diff(np.flatnonzero(support))>1)+1)
    groups=[g for g in groups if len(g)>=25 and d[g[0]]<200]
    if not groups:return {'method':'unresolved','segment_xy':None,'quality_score':quality}
    g=max(groups,key=len);a=xy[g[0]];b=xy[g[-1]]
    return {'method':'blur_axis_estimate','segment_xy':[a.tolist(),b.tolist()],
            'angle_degrees':float(np.degrees(angles[k])),'quality_score':quality,
            'endpoint_meaning':'Supported blur-axis extent; uncertain, not instantaneous shaft or physical tip.',
            'near_image_boundary':bool(min(b[0],b[1],frame.shape[1]-b[0],frame.shape[0]-b[1])<12)}


def draw_segment(image,result,color):
    if not result['segment_xy']:return
    a,b=np.array(result['segment_xy'])
    if result['method']=='blur_axis_estimate':
        for t in np.arange(0,1,.07):
            p=a+t*(b-a);q=a+min(t+.04,1)*(b-a)
            cv2.line(image,tuple(p.astype(int)),tuple(q.astype(int)),color,3,cv2.LINE_AA)
    else:cv2.line(image,tuple(a.astype(int)),tuple(b.astype(int)),color,3,cv2.LINE_AA)


def main():
    config=json.loads((ROOT/'strike_events.json').read_text())
    raw={r['frame']:r for r in map(json.loads,(ROOT/config['pose_data']).read_text().splitlines())}
    sequences=[]
    for event in config['strikes']:
        hit=event['contact_frame'];sequence=[]
        for frame in range(hit-config['pre_hit_frames'],hit+1):
            people={p['track_id']:p for p in raw[frame]['models']['rtmpose']['people']}
            sequence.append({'frame':frame,'strike':event['strike'],'offset_frames':frame-hit,
                'relative_time_ms':1000*(raw[frame]['time_seconds']-raw[hit]['time_seconds']),
                'attacker':people[event['attacker_track_id']],'receiver':people[event['receiver_track_id']]})
        sequences.append(sequence)
    data={'source':config['video'],'sequences':sequences}
    wanted={r['frame']:r for s in data['sequences'] for r in s}
    review=OUT/'shinai_review';review.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(str(ROOT/data['source']));lsd=cv2.createLineSegmentDetector(0)
    observations=[];thumbnails={};frame_index=0
    # Import only color values; the plotting module does not run inference.
    from plot_strikes import FRAME_COLORS
    while True:
        ok,image=cap.read()
        if not ok:break
        if frame_index in wanted:
            row=wanted[frame_index]
            if row['offset_frames']==-15:
                reference=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32),(0,0),5)
            lines=lsd.detect(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))[0]
            lines=np.empty((0,4)) if lines is None else lines.reshape(-1,4)
            grip=np.array(row['attacker']['keypoints_xy'])[[9,10]].mean(axis=0)
            result=sharp_segments(image,grip,'attacker',lines)
            if result is None:result=blurred_axis(image,reference,grip)
            record={'frame':frame_index,'strike':row['strike'],'relative_time_ms':row['relative_time_ms'],'attacker':result}
            if row['offset_frames']==0:
                grip=np.array(row['receiver']['keypoints_xy'])[[9,10]].mean(axis=0)
                record['receiver']=sharp_segments(image,grip,'receiver',lines)
            observations.append(record)
            rgb=FRAME_COLORS[row['offset_frames']+15][:3];color=tuple(int(x*255) for x in rgb[::-1])
            canvas=image.copy();draw_segment(canvas,result,color)
            if record.get('receiver'):draw_segment(canvas,record['receiver'],(140,140,140))
            canvas=cv2.resize(canvas,(768,432))
            cv2.rectangle(canvas,(0,0),(768,29),(22,22,22),-1)
            cv2.putText(canvas,f"Frame {frame_index} | {row['relative_time_ms']:.0f} ms | {result['method']}",(8,21),0,.55,(255,255,255),1)
            cv2.imwrite(str(review/f'frame_{frame_index:06d}.jpg'),canvas)
            thumbnails.setdefault(row['strike'],[]).append(canvas)
        frame_index+=1
    cap.release()
    for strike,images in thumbnails.items():
        sheet=np.vstack([np.hstack(images[j:j+4]) for j in range(0,16,4)])
        cv2.imwrite(str(review/f'strike_{strike}_all_frames.jpg'),sheet)
    counts=dict(Counter(r['attacker']['method'] for r in observations))
    result={'source':data['source'],'frame_count':len(observations),'counts':counts,
        'method':'Wrist-gated LSD + bamboo-color filtering; pre-strike difference-image ray search for blurred frames.',
        'scope':'Heuristic tuned for this stationary-camera, left-to-right drill, not a general shinai detector.',
        'limitations':['Blur axes describe exposure-averaged motion, not a unique instantaneous shinai position.',
                      'Segment endpoints are image-supported extent, not validated handle/tip coordinates.',
                      'Quality scores are heuristic, not probabilities. No ground-truth accuracy measurement.',
                      'No interpolation. Each frame is independently estimated using its image and wrist positions.'],
        'observations':observations}
    (OUT/'shinai_detections.json').write_text(json.dumps(result,indent=2))
    with (OUT/'shinai_detections.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['strike','frame','relative_time_ms','method','x1','y1','x2','y2','angle_degrees','quality_score'])
        for r in observations:
            q=r['attacker'];points=q['segment_xy'] or [[None,None],[None,None]]
            writer.writerow([r['strike'],r['frame'],r['relative_time_ms'],q['method'],*points[0],*points[1],q.get('angle_degrees'),q['quality_score']])
    print(counts,flush=True)
    assert len(observations)==80


if __name__=='__main__':main()
