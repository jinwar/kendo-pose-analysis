"""Plot every pre-contact frame using saved YOLOX + RTMPose observations only.

Contact frames and within-strike identities are reviewed annotations, not inferred
from confidence or wrist proximity. No pose smoothing/interpolation is applied.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable

EDGES=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),
       (11,13),(13,15),(12,14),(14,16),(0,1),(0,2),(1,3),(2,4)]
BLUE='#1379bb'
GRAY='#929aa5'
SLOT=1600
ANCHOR=1275


def skeleton(ax,person,origin,slot,threshold,color,lw=1.1,y_shift=0):
    xy=np.array(person['keypoints_xy'],dtype=float)
    xy[:,1]+=y_shift
    score=np.array(person['keypoint_scores'])
    xy[:,0]+=slot*SLOT+ANCHOR-origin
    valid=(score>=threshold)&np.isfinite(xy).all(axis=1)
    for a,b in EDGES:
        if valid[a] and valid[b]:
            ax.plot(xy[[a,b],0],xy[[a,b],1],color=color,lw=lw,solid_capstyle='round')
    ax.scatter(xy[valid,0],xy[valid,1],s=4,color=color,zorder=3)


FRAME_COLORS=plt.get_cmap('turbo')(np.linspace(.06,.94,16))
FRAME_CMAP=ListedColormap(FRAME_COLORS)
FRAME_NORM=BoundaryNorm(np.arange(-15.5,1,1),16)


def shinai_segment(ax,detection,origin,y_shift,color):
    if not detection or not detection.get('segment_xy'):return
    xy=np.array(detection['segment_xy'],dtype=float)
    xy[:,0]+=ANCHOR-origin
    xy[:,1]+=y_shift
    style='--' if detection['method']=='blur_axis_estimate' else '-'
    ax.plot(xy[:,0],xy[:,1],color=color,lw=2.1,ls=style,solid_capstyle='round',zorder=4)


def draw_sequence(ax,sequence,origin,threshold,frame_labels=True):
    receiver=sequence[-1]['receiver']
    y_shift=650-float(np.mean(np.array(receiver['keypoints_xy'])[[11,12],1]))
    # The receiver is drawn exactly once, using its contact-frame posture.
    skeleton(ax,receiver,origin,0,threshold,'#687480',lw=2,y_shift=y_shift)
    shinai_segment(ax,sequence[-1].get('receiver_shinai'),origin,y_shift,'#687480')
    for j,row in enumerate(sequence):
        person=row['attacker']
        if person is not None:
            skeleton(ax,person,origin,0,threshold,FRAME_COLORS[j],lw=1.15,y_shift=y_shift)
            shinai_segment(ax,row.get('shinai'),origin,y_shift,FRAME_COLORS[j])
    ax.set_xlim(-100,1530);ax.set_ylim(1140,-30);ax.set_aspect('equal')
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)


def time_bar(fig,axes):
    bar=fig.colorbar(ScalarMappable(norm=FRAME_NORM,cmap=FRAME_CMAP),ax=axes,
                    fraction=.025,pad=.025,shrink=.72,ticks=[-15,-12,-9,-6,-3,0])
    bar.ax.set_yticklabels(['−500','−400','−300','−200','−100','0 · hit'])
    bar.set_label('Time relative to contact (ms) · one color per frame',fontsize=10,labelpad=12)
    bar.outline.set_visible(False)


def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=220,facecolor='white',bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'),facecolor='white',bbox_inches='tight')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events',type=Path,default=ROOT/'strike_events.json')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/strike_analysis')
    args=parser.parse_args()
    config=json.loads(args.events.read_text());out=args.output;out.mkdir(parents=True,exist_ok=True)
    shinai_path=out/'shinai_detections.json'
    shinai_data=json.loads(shinai_path.read_text()) if shinai_path.exists() else None
    shinai_by_frame={r['frame']:r for r in shinai_data['observations']} if shinai_data else {}
    if shinai_data and shinai_data['source']!=config['video']:
        raise ValueError('Shinai source does not match selected video')
    rows={r['frame']:r for r in map(json.loads,(ROOT/config['pose_data']).read_text().splitlines())}
    sequences=[];origins=[];audit=[]
    for event in config['strikes']:
        hit=event['contact_frame'];t0=rows[hit]['time_seconds'];sequence=[]
        for frame in range(hit-config['pre_hit_frames'],hit+1):
            row=rows[frame];people={p['track_id']:p for p in row['models'][config['model']]['people']}
            item={'strike':event['strike'],'frame':frame,'time_seconds':row['time_seconds'],
                  'offset_frames':frame-hit,'relative_time_ms':1000*(row['time_seconds']-t0)}
            for role in ['attacker','receiver']:
                item[role]=people.get(event[role+'_track_id'])
            if shinai_data:
                detection=shinai_by_frame[frame]
                if detection['strike']!=event['strike']:raise ValueError('Shinai strike mismatch')
                item['shinai']=detection['attacker']
                if detection.get('receiver'):item['receiver_shinai']=detection['receiver']
            sequence.append(item)
        receiver=sequence[-1]['receiver']
        if receiver is None:raise ValueError('Receiver missing at selected contact frame')
        # A single translation per strike, fixed for every frame. No per-frame
        # recentering, scaling, mirroring, or time warping; retain actual approach.
        origin=float(np.mean(np.array(receiver['keypoints_xy'])[[11,12],0]))
        sequences.append(sequence);origins.append(origin)
        audit.append(dict(event,contact_time_seconds=t0,
             first_frame=sequence[0]['frame'],last_frame=hit,
             missing_attacker_frames=sum(x['attacker'] is None for x in sequence),
             missing_receiver_frames=sum(x['receiver'] is None for x in sequence),
             receiver_hip_x_at_contact=origin))
    fig,axes=plt.subplots(5,1,figsize=(7.6,18))
    fig.subplots_adjust(left=.09,right=.87,top=.91,bottom=.055,hspace=.24)
    for i,(ax,sequence,origin,event) in enumerate(zip(axes,sequences,origins,audit)):
        draw_sequence(ax,sequence,origin,config['keypoint_threshold'])
        ax.set_title(f"Strike {i+1}  |  Contact {event['contact_time_seconds']:.3f} s",loc='left',fontsize=12,pad=5)
    fig.suptitle('Five strikes · posture and shinai' if shinai_data else 'Five strikes · overlaid pre-contact postures',x=.09,y=.985,ha='left',fontsize=15,fontweight='bold')
    fig.text(.09,.952,'Left attacker: 16 consecutive frames, −0.5 s to contact.\nRight receiver: contact-frame posture, drawn once in gray.',fontsize=10,color='#52606d',linespacing=1.7,va='top')
    time_bar(fig,axes)
    for ax in axes:ax.set_anchor('W')
    note=('Thicker lines: shinai segments · dashed: uncertain blur-axis estimates\nEndpoints are visible support, not measured tips. Contact ≈ ±1 frame.' if shinai_data else 'YOLOX + RTMPose · same scale · fixed receiver hip reference\nNo smoothing or interpolation. Visual contact ≈ ±1 frame; hidden joints are estimates.')
    fig.text(.09,.02,note,fontsize=9,color='#52606d')
    save(fig,out/'five_strikes_aligned')
    for event,sequence,origin in zip(audit,sequences,origins):
        fig,ax=plt.subplots(figsize=(9,6.7))
        fig.subplots_adjust(left=.04,right=.86,top=.82,bottom=.12)
        draw_sequence(ax,sequence,origin,config['keypoint_threshold'])
        fig.suptitle(f"Strike {event['strike']} · contact at {event['contact_time_seconds']:.3f} s",x=.04,ha='left',fontsize=17,fontweight='bold')
        fig.text(.04,.88,'Left attacker: one color per frame · Right receiver: one fixed gray posture',fontsize=10,color='#52606d')
        time_bar(fig,ax)
        fig.text(.04,.04,note,fontsize=9,color='#52606d')
        save(fig,out/f"strike_{event['strike']}_frames")
    export={'source':config['video'],'pipeline':'YOLOX-m + RTMPose-m','contact_method':config['contact_method'],
       'coordinate_system':'Original image pixels; plot translations are recorded separately; image y increases downwards.',
       'plot_method':'Overlay all attacker frames; receiver contact pose drawn once. Fixed translation per strike places receiver contact hip at (1275,650). No framewise recentering.',
       'shinai_summary':{'counts':shinai_data['counts'],'limitations':shinai_data['limitations']} if shinai_data else None,
       'threshold':config['keypoint_threshold'],'events':audit,'sequences':sequences}
    (out/'aligned_poses.json').write_text(json.dumps(export,indent=2))
    with (out/'aligned_keypoints.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['strike','frame','offset_frames','relative_time_ms','role','track_id','joint_index','x_px','y_px','score'])
        for sequence in sequences:
            for row in sequence:
                for role in ['attacker','receiver']:
                    p=row[role]
                    if p:
                        for j,(xy,score) in enumerate(zip(p['keypoints_xy'],p['keypoint_scores'])):
                            writer.writerow([row['strike'],row['frame'],row['offset_frames'],row['relative_time_ms'],role,p['track_id'],j,*xy,score])
    cards=''.join(f'<h2>Strike {i}</h2><a href="strike_{i}_frames.svg">Vector figure</a> · <a href="shinai_review/strike_{i}_all_frames.jpg">Check detections against all 16 video frames</a><img src="strike_{i}_frames.png" alt="Strike {i} postures and shinai">' for i in range(1,6))
    shinai_note=(f"<p><strong>Shinai prototype:</strong> {shinai_data['counts'].get('edge_supported',0)} frames use visible edge segments (solid); {shinai_data['counts'].get('blur_axis_estimate',0)} use uncertain motion-blur axes (dashed)." ' Color matches the posture from the same frame. These are image-supported shaft segments, not measured shinai tips or full lengths. The fixed receiver’s contact-frame shinai is gray. No temporal interpolation.</p><p><a href="shinai_detections.csv">Shinai CSV</a> · <a href="shinai_detections.json">Methods and estimates JSON</a></p>' if shinai_data else '')
    (out/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Five aligned strikes</title>
<style>body{font:16px system-ui;max-width:1800px;margin:32px auto;padding:20px;color:#243344}img{width:100%;margin:20px 0}a{color:#1379bb}p{line-height:1.6}</style>
<h1>Five strikes aligned to contact</h1><p>20261108_120241.mp4 · YOLOX + RTMPose only. Each overview row is one strike, with every consecutive frame from −15 to 0. Contact is visually selected (approximately ±1 frame), not automatically detected. Attacker postures are overlaid in one shared coordinate system, with a different color for every frame. The receiver is drawn once in gray at the contact-frame position. The colorbar shows time before contact.</p>
<a href="five_strikes_aligned.png">Full-resolution overview PNG</a> · <a href="five_strikes_aligned.svg">Overview SVG</a> · <a href="aligned_keypoints.csv">Keypoints CSV</a>
<img src="five_strikes_aligned.png" alt="Five rows of aligned pre-contact skeletons">'''+shinai_note+cards+'</html>')
    assert len(sequences)==5 and all(len(s)==16 for s in sequences)
    assert all(s[-1]['relative_time_ms']==0 and all(a['frame']+1==b['frame'] for a,b in zip(s,s[1:])) for s in sequences)
    print(json.dumps(audit,indent=2));print('Figures saved:',out)


if __name__=='__main__':main()
