"""Contact-aligned robust, confidence-weighted average of the five observed strikes.

No phase warping, scale normalization, smoothing, or invented shinai tips.
Spread measures between-strike variation plus tracking error, not a confidence
interval for the person's true movement. Image units are pixels, not metres.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from plot_strikes import plt, EDGES, FRAME_COLORS, time_bar, save
from matplotlib.patches import Ellipse

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/average_strike'


def weighted_position(points,confidence):
    points=np.asarray(points,float);confidence=np.asarray(confidence,float)
    valid=np.isfinite(points).all(axis=1)&np.isfinite(confidence)&(confidence>=.3)
    indices=np.flatnonzero(valid)
    if len(indices)<3:return None
    p=points[valid];w=np.clip(confidence[valid],0,1)
    # Conservative radial Huber reweighting: flag/downweight, do not silently
    # delete a strike that may represent genuine motion variation.
    center=np.median(p,axis=0);distance=np.linalg.norm(p-center,axis=1)
    median=np.median(distance);mad=np.median(np.abs(distance-median))
    threshold=max(75.,median+3*1.4826*mad)
    robust=np.minimum(1,threshold/np.maximum(distance,1e-9));w*=robust
    mean=np.average(p,axis=0,weights=w)
    delta=p-mean;denominator=w.sum()-np.sum(w*w)/w.sum()
    covariance=(delta.T*w)@delta/denominator if denominator>0 else np.zeros((2,2))
    return {'mean_xy':mean.tolist(),'covariance_px2':covariance.tolist(),
            'n':len(indices),'effective_n':float(w.sum()**2/np.sum(w*w)),
            'used_strike_indices':indices.tolist(),'weights':w.tolist(),
            'downweighted_strike_indices':indices[robust<1].tolist()}


def axial_mean(angles,weights):
    angles=np.asarray(angles);weights=np.asarray(weights)
    z=np.sum(weights*np.exp(2j*angles))/np.sum(weights)
    angle=float(np.angle(z)/2)
    return angle,float(abs(z))


def average_shinai(observations,translations):
    anchors=[];angles=[];weights=[];methods=[]
    for row,shift in zip(observations,translations):
        q=row.get('shinai');person=row.get('attacker')
        if not q or not q.get('segment_xy') or not person:continue
        wrist_scores=np.asarray(person['keypoint_scores'])[[9,10]]
        if np.min(wrist_scores)<.3:continue
        a,b=np.array(q['segment_xy']);u=b-a;length=np.linalg.norm(u)
        if length<1:continue
        u/=length
        if u[0]<0:u=-u
        grip=np.array(person['keypoints_xy'])[[9,10]].mean(axis=0)
        projected=a+u*np.dot(grip-a,u)
        anchors.append(projected+shift);angles.append(np.arctan2(u[1],u[0]))
        # Explicit heuristic reliability factors, not calibrated probabilities.
        factor=1 if q['method']=='edge_supported' else .25
        weights.append(float(np.clip(wrist_scores.mean(),0,1)*factor));methods.append(q['method'])
    if len(anchors)<3:return None
    weights=np.array(weights);angles=np.array(angles);anchors=np.array(anchors)
    theta,concentration=axial_mean(angles,weights)
    residual=np.angle(np.exp(2j*(angles-theta)))/2
    center=np.average(anchors,axis=0,weights=weights)
    return {'anchor_xy':center.tolist(),'angle_radians':theta,
            'angle_degrees':float(np.degrees(theta)),
            'angular_rms_spread_degrees':float(np.degrees(np.sqrt(np.average(residual**2,weights=weights)))),
            'axial_concentration':concentration,'n':len(anchors),
            'edge_supported_count':methods.count('edge_supported'),
            'blur_estimate_count':methods.count('blur_axis_estimate'),
            'weights':weights.tolist(),
            'all_blurred':all(m=='blur_axis_estimate' for m in methods),
            'anchor_definition':'Projection of each two-wrist midpoint onto its estimated shaft line, then averaged.',
            'display_length_px':300,'display_start_distance_px':60,
            'endpoint_meaning':'Schematic orientation guide only. Neither endpoint is a measured tip.'}


def draw_body(ax,joints,color,width=1.3,alpha=1):
    for a,b in EDGES:
        if joints[a] and joints[b]:
            xy=np.array([joints[a]['mean_xy'],joints[b]['mean_xy']])
            ax.plot(xy[:,0],xy[:,1],color=color,lw=width,alpha=alpha,solid_capstyle='round')
    xy=np.array([p['mean_xy'] for p in joints if p])
    if len(xy):ax.scatter(xy[:,0],xy[:,1],s=7,color=color,alpha=alpha,zorder=3)


def spread_ellipse(ax,joint,color):
    if not joint:return
    values,vectors=np.linalg.eigh(np.array(joint['covariance_px2']))
    values=np.maximum(values,0);major=vectors[:,-1]
    ax.add_patch(Ellipse(joint['mean_xy'],2*np.sqrt(values[-1]),2*np.sqrt(values[0]),
        angle=np.degrees(np.arctan2(major[1],major[0])),facecolor=color,edgecolor='none',alpha=.13,zorder=0))


def draw_shinai(ax,q,color,spread=False):
    if not q:return
    theta=q['angle_radians'];anchor=np.array(q['anchor_xy']);u=np.array([np.cos(theta),np.sin(theta)])
    xy=anchor+np.array([60,360])[:,None]*u
    ax.plot(xy[:,0],xy[:,1],color=color,lw=2,ls='-',zorder=4)
    if spread:
        delta=np.deg2rad(q['angular_rms_spread_degrees'])
        a=np.linspace(theta-delta,theta+delta,25)
        fan=anchor+360*np.stack([np.cos(a),np.sin(a)],axis=1)
        polygon=np.vstack([anchor,fan,anchor]);ax.fill(polygon[:,0],polygon[:,1],color=color,alpha=.08,lw=0,zorder=0)


def setup_axis(ax):
    ax.set_xlim(-1450,280);ax.set_ylim(560,-730);ax.set_aspect('equal');ax.axis('off')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'outputs/strike_analysis/aligned_poses.json')
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--max-time-span-ms',type=float,default=5,help='Maximum permitted within-offset timestamp span; stored in output')
    args=parser.parse_args();source=args.input.resolve();out=args.output
    out.mkdir(parents=True,exist_ok=True)
    data=json.loads(source.read_text());sequences=data['sequences']
    if len(sequences)!=5:raise ValueError('This analysis expects the five reviewed strikes')
    offsets=[[r['offset_frames'] for r in s] for s in sequences]
    if not all(o==offsets[0] for o in offsets):raise ValueError('Frame offsets differ between strikes')
    times=np.array([[r['relative_time_ms'] for r in s] for s in sequences])
    timing_span=float(np.ptp(times,axis=0).max())
    if timing_span>args.max_time_span_ms:raise ValueError(f'Frame timing span {timing_span:.3f} ms exceeds permitted {args.max_time_span_ms} ms')
    translations=[]
    for s in sequences:
        p=s[-1]['receiver'];hip=np.array(p['keypoints_xy'])[[11,12]]
        if min(p['keypoint_scores'][11:13])<.3:raise ValueError('Contact receiver hip is unreliable')
        translations.append(-hip.mean(axis=0))
    translations=np.array(translations)
    receiver=[]
    for joint in range(17):
        receiver.append(weighted_position([np.array(s[-1]['receiver']['keypoints_xy'][joint])+shift for s,shift in zip(sequences,translations)],
            [s[-1]['receiver']['keypoint_scores'][joint] for s in sequences]))
    averages=[]
    for column,offset in enumerate(offsets[0]):
        observations=[s[column] for s in sequences];joints=[]
        for j in range(17):
            positions=[];scores=[]
            for row,shift in zip(observations,translations):
                p=row['attacker']
                positions.append(np.array(p['keypoints_xy'][j])+shift if p else [np.nan,np.nan])
                scores.append(p['keypoint_scores'][j] if p else 0)
            joints.append(weighted_position(positions,scores))
        averages.append({'offset_frames':offset,'relative_time_ms':float(times[:,column].mean()),
            'source_frames':[r['frame'] for r in observations],'joints':joints,
            'shinai':average_shinai(observations,translations)})
    out.mkdir(parents=True,exist_ok=True)
    metadata={'source_video':data['source'],'input':str(source.relative_to(ROOT)),'strikes':5,
        'translations_xy':translations.tolist(),'max_time_alignment_span_ms':timing_span,'permitted_time_span_ms':args.max_time_span_ms,
        'coordinate_system':'Image pixels relative to receiver contact-frame hip midpoint (0,0); y down.',
        'method':'Confidence-weighted joint mean, confidence>=0.3, clipped weights<=1, radial Huber downweight beyond max(75px, median distance + 3 scaled MAD). At least 3 observations required.',
        'spread':'Weighted sample covariance ellipses with one-SD semi-axes; descriptive scatter, NOT confidence intervals. Includes real variation and detector error.',
        'shinai_method':'Axial circular orientation mean and mean line anchor projected from wrists; weights = wrist confidence times 1.0 for edge segments or 0.25 for blur estimates. These factors are heuristic.',
        'limitations':['Five strikes from one recording only; not an anatomical or physical model.',
            'Contact times manually chosen, approximately +/-1 frame. Timing uncertainty is not included in ellipses.',
            'No phase/time warping, temporal smoothing, bone-length constraint, rotation, or scale normalization.',
            'Averaging cannot remove shared pose-estimation bias under armor; means may shorten apparent limbs.',
            'Shinai guides have fixed display length: no tip position, full weapon length, or tip speed is inferred.'],
        'receiver_contact_mean':receiver,'frames':averages}
    (out/'average_strike.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))
    with (out/'average_joints.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['offset_frames','time_ms','joint_index','x_px','y_px','n','effective_n','sd_x_px','sd_y_px','cov_xy_px2','downweighted_strikes_1based'])
        for row in averages:
            for j,p in enumerate(row['joints']):
                if p:
                    cov=np.array(p['covariance_px2']);writer.writerow([row['offset_frames'],row['relative_time_ms'],j,*p['mean_xy'],p['n'],p['effective_n'],*np.sqrt(np.diag(cov)),cov[0,1],','.join(str(i+1) for i in p['downweighted_strike_indices'])])
    # Main view: mean sequence in the same coordinate system as the original overlays.
    fig,ax=plt.subplots(figsize=(10,9));fig.subplots_adjust(left=.04,right=.86,top=.85,bottom=.16)
    draw_body(ax,receiver,'#76818b',width=2)
    for i,row in enumerate(averages):
        draw_body(ax,row['joints'],FRAME_COLORS[i]);draw_shinai(ax,row['shinai'],FRAME_COLORS[i])
    setup_axis(ax);time_bar(fig,ax)
    fig.suptitle('Typical strike · average of five repetitions',x=.04,y=.98,ha='left',fontsize=19,fontweight='bold')
    fig.text(.04,.9,'Contact-aligned, confidence-weighted mean · 16 frames from −0.5 s to contact\nGray: average receiver at contact. Colors: attacker and shinai orientation.',fontsize=11,color='#52606d')
    fig.text(.04,.035,'Shinai lines are fixed-length orientation guides, not measured tips. All shinai lines are solid; blurred inputs remain downweighted.\nNo phase warping or temporal smoothing. Five-strike average includes tracking errors; contact ≈ ±1 frame.',fontsize=10,color='#52606d')
    save(fig,out/'average_strike')
    # Every frame separately, with descriptive joint scatter and orientation spread.
    fig,axes=plt.subplots(4,4,figsize=(17,17));fig.subplots_adjust(left=.025,right=.99,top=.9,bottom=.075,hspace=.25,wspace=.03)
    for i,(ax,row) in enumerate(zip(axes.flat,averages)):
        draw_body(ax,receiver,'#a2a9b0',width=1)
        for p in row['joints']:spread_ellipse(ax,p,FRAME_COLORS[i])
        draw_body(ax,row['joints'],FRAME_COLORS[i],width=1.7);draw_shinai(ax,row['shinai'],FRAME_COLORS[i],spread=True)
        setup_axis(ax);ax.set_title(f"{row['relative_time_ms']:.0f} ms"+(' · contact' if i==15 else ''),fontsize=12)
    fig.suptitle('Average strike · frame by frame',x=.025,y=.98,ha='left',fontsize=23,fontweight='bold')
    fig.text(.025,.94,'Read left to right, then down. Shading shows between-strike scatter (one-SD axes), not measurement confidence.',fontsize=12,color='#52606d')
    fig.text(.025,.025,'Five repetitions · low-confidence joints excluded; unusually distant estimates downweighted. No time warping.\nShinai fan: weighted angular RMS spread. Shaft guide length is schematic; all guides use solid lines.\nVariation includes genuine strike differences, pose errors, and alignment errors. Contact uncertainty ≈ ±1 frame is not propagated.',fontsize=11,color='#52606d')
    save(fig,out/'average_frames_spread')
    # Direct audit: original translated poses vs average at six shared instants.
    fig,axes=plt.subplots(2,3,figsize=(14,10));fig.subplots_adjust(left=.025,right=.99,top=.87,bottom=.07,hspace=.2)
    for ax,i in zip(axes.flat,[0,3,6,9,12,15]):
        for s,shift in zip(sequences,translations):
            p=s[i]['attacker']
            if p:
                joints=[{'mean_xy':(np.array(x)+shift).tolist()} if score>=.3 else None for x,score in zip(p['keypoints_xy'],p['keypoint_scores'])]
                draw_body(ax,joints,'#c8cdd3',width=.8,alpha=.65)
        draw_body(ax,averages[i]['joints'],'#1479b8',width=2)
        draw_body(ax,receiver,'#76818b',width=1)
        setup_axis(ax);ax.set_title(f"{averages[i]['relative_time_ms']:.0f} ms",fontsize=12)
    fig.suptitle('Five observed poses and their average',x=.025,y=.98,ha='left',fontsize=21,fontweight='bold')
    fig.text(.025,.925,'Light gray: each translated strike · Blue: confidence-weighted mean · Dark gray: fixed average receiver',fontsize=11,color='#52606d')
    save(fig,out/'average_vs_observations')
    (out/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Average kendo strike</title>
<style>body{font:16px system-ui;max-width:1300px;margin:32px auto;padding:20px;color:#243344}img{max-width:100%;margin:20px 0}p{line-height:1.6}a{color:#1379bb}</style>
<h1>Typical strike — average of five repetitions</h1>
<p>Contact-aligned average of the same five strikes. Each strike is translated once using the receiver's hip at contact; timing, pixel scale, and approach are preserved. Low-confidence joints are excluded, and unusually distant estimates receive reduced weight. No phase warping or temporal smoothing.</p>
<p>Shinai orientation and line position are averaged with blurred estimates weighted at ¼ of clear estimates. Drawn shaft lengths are schematic, not measurements of weapon length or tip location. All guides are solid; detection methods and weights remain available in the JSON.</p>
<a href="average_strike.png">Overview PNG</a> · <a href="average_strike.svg">Overview SVG</a> · <a href="average_joints.csv">Joint statistics CSV</a> · <a href="average_strike.json">Full method and data</a>
<img src="average_strike.png" alt="Average strike with color coded postures">
<h2>Every averaged frame and its variation</h2><p>Ellipses show one-SD covariance axes across the five estimates; shaded fans show angular RMS spread. These are descriptive variation, not confidence intervals. Genuine strike differences and detector errors are mixed together. Manual contact timing uncertainty of approximately one frame is not included.</p>
<a href="average_frames_spread.svg">Frame-by-frame SVG</a><img src="average_frames_spread.png" alt="Sixteen average poses with variation">
<h2>Check the average against the observations</h2><img src="average_vs_observations.png" alt="Individual poses and weighted average">
<p>This represents these five recorded strikes, not a definitive technique model. <a href="../strike_analysis/index.html">Return to the five individual strikes</a>.</p></html>''')
    page=out/'index.html'
    import os
    individual=os.path.relpath(source.parent/'index.html',out)
    page.write_text(page.read_text().replace('../strike_analysis/index.html',individual))
    counts=[p['n'] for r in averages for p in r['joints'] if p]
    downweighted=sum(len(p['downweighted_strike_indices']) for r in averages for p in r['joints'] if p)
    print(f'Saved {len(averages)} mean poses; contributors per joint {min(counts)}–{max(counts)}; {downweighted} joint observations downweighted.')
    print(f'Max matching-frame time span: {timing_span:.6f} ms')
    page=out/'index.html'
    page.write_text(page.read_text().replace('<h1>',f'<p>Maximum timestamp spread at matching frame offsets: {timing_span:.3f} ms. Original frames retained without resampling.</p><h1>',1))
    print('Output:',out/'index.html')
    from results_index import build_index
    build_index(ROOT/'outputs')


if __name__=='__main__':main()
