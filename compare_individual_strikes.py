"""Pair the two videos' individual strikes in chronological order, without averaging."""
import json
from pathlib import Path
import numpy as np
from plot_strikes import plt, draw_sequence, time_bar, save
from average_strikes import draw_body, setup_axis

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/strike_comparison'
COLORS=['#1479b8','#dd5729']


def origin(sequence):
    receiver=next(r for r in sequence if r['offset_frames']==0)['receiver']
    if receiver is None or min(receiver['keypoint_scores'][11:13])<.3:
        raise ValueError('Reliable receiver contact hips required')
    return np.array(receiver['keypoints_xy'])[[11,12]].mean(axis=0)


def draw_frame(ax,row,anchor,color,threshold):
    person=row['attacker']
    if person:
        joints=[{'mean_xy':(np.array(xy)-anchor).tolist()} if score>=threshold else None
                for xy,score in zip(person['keypoints_xy'],person['keypoint_scores'])]
        draw_body(ax,joints,color,width=1.5,alpha=.85)
    q=row.get('shinai')
    if q and q.get('segment_xy'):
        xy=np.array(q['segment_xy'])-anchor
        ax.plot(xy[:,0],xy[:,1],color=color,lw=2,ls='-',zorder=4)


def main():
    paths=[ROOT/'outputs/strike_analysis/aligned_poses.json',ROOT/'outputs/strike_analysis_second/aligned_poses.json']
    data=[json.loads(p.read_text()) for p in paths]
    if any(len(d['sequences'])!=5 for d in data):raise ValueError('Five strikes per video required')
    OUT.mkdir(parents=True,exist_ok=True)
    audit=[];cards=[]
    overview,axes=plt.subplots(5,2,figsize=(16,23))
    overview.subplots_adjust(left=.035,right=.9,top=.94,bottom=.055,hspace=.28,wspace=.05)
    for k,(a,b) in enumerate(zip(data[0]['sequences'],data[1]['sequences']),1):
        pair=[a,b];anchors=[origin(s) for s in pair]
        offsets=[r['offset_frames'] for r in a]
        if offsets!=list(range(-15,1)) or offsets!=[r['offset_frames'] for r in b]:
            raise ValueError('Matching 16-frame windows required')
        mismatch=max(abs(x['relative_time_ms']-y['relative_time_ms']) for x,y in zip(a,b))
        if mismatch>10:raise ValueError('Matching frame times differ by over 10 ms')
        audit.append({'pair':k,'source_strikes':[s[-1]['strike'] for s in pair],
                      'source_frames':[[r['frame'] for r in s] for s in pair],
                      'relative_times_ms':[[r['relative_time_ms'] for r in s] for s in pair],
                      'receiver_contact_hips_xy':[x.tolist() for x in anchors],
                      'max_time_difference_ms':mismatch})
        fig,panels=plt.subplots(1,2,figsize=(16,8))
        fig.subplots_adjust(left=.025,right=.9,top=.84,bottom=.14,wspace=.06)
        for j,(sequence,anchor) in enumerate(zip(pair,anchors)):
            title=f"Person {'AB'[j]} · strike {k} · contact {sequence[-1]['time_seconds']:.3f} s"
            for ax in [panels[j],axes[k-1,j]]:
                draw_sequence(ax,sequence,anchor[0],data[j]['threshold'])
                ax.set_title(title,fontsize=13,loc='left')
        fig.suptitle(f'Strike {k} · individual comparison',fontsize=23,fontweight='bold',x=.025,ha='left')
        fig.text(.025,.89,'Original poses · matching frame colors · receiver fixed at contact · no averaging',fontsize=12)
        fig.text(.025,.04,f'Solid shinai: detected shaft segments, not verified tips. Maximum paired timestamp difference: {mismatch:.3f} ms.\nSame pixel scale and receiver-hip reference; no body-size normalization or time warping. Contact ≈ ±1 frame.',fontsize=11,color='#52606d')
        time_bar(fig,panels);save(fig,OUT/f'strike_{k}_comparison')
        fig,grid=plt.subplots(4,4,figsize=(17,17))
        fig.subplots_adjust(left=.025,right=.99,top=.9,bottom=.075,hspace=.25,wspace=.03)
        for i,ax in enumerate(grid.flat):
            for j,(sequence,anchor,color) in enumerate(zip(pair,anchors,COLORS)):
                draw_frame(ax,sequence[i],anchor,color,data[j]['threshold'])
            setup_axis(ax)
            ax.set_title(f"A {a[i]['relative_time_ms']:.0f} / B {b[i]['relative_time_ms']:.0f} ms"+(' · contact' if i==15 else ''),fontsize=11)
        fig.suptitle(f'Strike {k} · original postures frame by frame',fontsize=22,fontweight='bold',x=.025,ha='left')
        fig.text(.025,.94,'Blue: Person A / video 1    Orange: Person B / video 2    ·    Read left to right, then down',fontsize=13)
        fig.text(.025,.025,'Each strike is translated once to its receiver’s contact hip. No averaging, interpolation or body-size correction.\nSolid shinai segments include uncertain blur estimates; endpoints are not verified tips. Manual contact ≈ ±1 frame.',fontsize=11,color='#52606d')
        save(fig,OUT/f'strike_{k}_matched_frames')
        cards.append(f'<h2>Strike {k} versus strike {k}</h2><p><a href="strike_{k}_comparison.svg">Side-by-side SVG</a> · <a href="strike_{k}_matched_frames.png">16-frame overlay PNG</a> · <a href="strike_{k}_matched_frames.svg">16-frame overlay SVG</a></p><img src="strike_{k}_comparison.png" alt="Strike {k} paired comparison"><details><summary>Show all 16 blue/orange frame overlays</summary><img src="strike_{k}_matched_frames.png" alt="Strike {k} original frames"></details>')
    overview.suptitle('Strike by strike · Person A (left) and Person B (right)',fontsize=23,fontweight='bold',x=.035,ha='left')
    overview.text(.035,.96,'Pairs follow recording order · 16 original poses per strike · matching time colors · no averaging',fontsize=13)
    overview.text(.035,.018,'Solid shinai segments include blur estimates; endpoints are not verified tips. Gray receiver: fixed contact pose.\nSame pixel scale and receiver-hip alignment; no body-size or perspective correction. Contact ≈ ±1 frame.',fontsize=12,color='#52606d')
    time_bar(overview,axes);save(overview,OUT/'five_strike_pairs')
    (OUT/'comparison.json').write_text(json.dumps({'inputs':[str(p.relative_to(ROOT)) for p in paths],
        'pairing':'Chronological: strike 1 vs 1 through strike 5 vs 5; not similarity-selected.',
        'method':'Original per-frame poses and detected shinai segments. Fixed translation by each receiver contact hip; no averaging, scaling or interpolation.',
        'pairs':audit},indent=2))
    (OUT/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Strike-by-strike comparison</title>
<style>body{font:16px system-ui;max-width:1500px;margin:32px auto;padding:20px;color:#243344}img{width:100%}p{line-height:1.6}summary{cursor:pointer}</style>
<h1>Strike-by-strike comparison</h1><p>Person A: attacker in video 1 (20261108_120241.mp4). Person B: attacker in video 2 (20261108_120335.mp4). Pairs follow recording order: 1 versus 1 through 5 versus 5. These are the original observations, with no averaging.</p>
<p>Each pair shares pixel scale, contact-frame offset and receiver-hip reference. All shinai lines are solid, including uncertain blur estimates. Segments show detected support, not full weapon length or verified tips. Body size and perspective are not corrected. Matching timestamps differ by less than 10 ms; actual times are shown in each frame panel. Manual contact uncertainty is approximately one frame.</p>
<p><a href="five_strike_pairs.png">Five-row overview PNG</a> · <a href="five_strike_pairs.svg">Overview SVG</a> · <a href="comparison.json">Pairing and timing audit</a></p>'''+''.join(cards)+'</html>')
    from results_index import build_index
    build_index(ROOT/'outputs')
    print('Saved five individual comparisons and 80 paired-frame panels:',OUT/'index.html')
    print('Maximum timestamp difference:',max(p['max_time_difference_ms'] for p in audit))


if __name__=='__main__':main()
