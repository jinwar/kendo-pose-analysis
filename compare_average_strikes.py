"""Compare the two video attackers at matched times relative to contact."""
import json
from pathlib import Path
import numpy as np
from average_strikes import draw_body, draw_shinai, setup_axis
from plot_strikes import plt, FRAME_COLORS, time_bar, save

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/average_comparison'


def main():
    paths=[ROOT/'outputs/average_strike/average_strike.json',ROOT/'outputs/average_strike_second/average_strike.json']
    data=[json.loads(p.read_text()) for p in paths]
    labels=['Person A · video 1','Person B · video 2']
    colors=['#1479b8','#dd5729']
    assert [r['offset_frames'] for r in data[0]['frames']]==[r['offset_frames'] for r in data[1]['frames']]
    delta=max(abs(a['relative_time_ms']-b['relative_time_ms']) for a,b in zip(data[0]['frames'],data[1]['frames']))
    if delta>5:raise ValueError('Comparison requires explicit time resampling')
    OUT.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(16,8))
    fig.subplots_adjust(left=.02,right=.9,top=.83,bottom=.14,wspace=.08)
    for ax,d,label in zip(axes,data,labels):
        draw_body(ax,d['receiver_contact_mean'],'#929aa5',width=1.8)
        for i,r in enumerate(d['frames']):
            draw_body(ax,r['joints'],FRAME_COLORS[i]);draw_shinai(ax,r['shinai'],FRAME_COLORS[i])
        setup_axis(ax);ax.set_title(label,fontsize=17)
    time_bar(fig,axes)
    fig.suptitle('Two people · average strike comparison',fontsize=23,fontweight='bold',x=.02,ha='left')
    fig.text(.02,.89,'Five strikes each · 16 consecutive frames · same time colors, image-pixel scale and receiver-hip origin',fontsize=12)
    fig.text(.02,.04,'Solid shinai lines show orientation only; lengths are schematic. Gray: each video’s average receiver at contact.\nNo time warping or body-size normalization. Camera perspective, body size and detector errors affect comparison.',fontsize=11,color='#52606d')
    save(fig,OUT/'two_average_strikes')
    fig,axes=plt.subplots(4,4,figsize=(17,17))
    fig.subplots_adjust(left=.025,right=.99,top=.9,bottom=.075,hspace=.25,wspace=.03)
    for i,ax in enumerate(axes.flat):
        for d,color in zip(data,colors):
            r=d['frames'][i]
            draw_body(ax,r['joints'],color,width=1.4,alpha=.85);draw_shinai(ax,r['shinai'],color)
        setup_axis(ax);ax.set_title(f"{data[0]['frames'][i]['relative_time_ms']:.0f} ms"+(' · contact' if i==15 else ''))
    fig.suptitle('Mean postures at matching times',fontsize=23,fontweight='bold',x=.025,ha='left')
    fig.text(.025,.94,'Blue: Person A / video 1    Orange: Person B / video 2    ·    Read left to right, then down',fontsize=13)
    fig.text(.025,.025,'Each mean is aligned to its receiver’s contact hip. Identical pixel axes; no body-size or perspective correction.\nFive strikes per person, manual contact ≈ ±1 frame. Shinai lines are solid orientation guides, not measured tips.',fontsize=11,color='#52606d')
    save(fig,OUT/'matched_mean_postures')
    (OUT/'comparison.json').write_text(json.dumps({'inputs':[str(p.relative_to(ROOT)) for p in paths],'labels':labels,'max_matching_time_difference_ms':delta,'alignment':'Receiver contact hip origin; original image pixels; no scaling or time warping.'},indent=2))
    (OUT/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Two-person strike comparison</title>
<style>body{font:16px system-ui;max-width:1500px;margin:32px auto;padding:20px;color:#243344}img{width:100%}p{line-height:1.6}</style>
<h1>Two-person average strike comparison</h1><p>Person A is the attacker in 20261108_120241.mp4; Person B is the attacker in 20261108_120335.mp4. Each average uses five visually reviewed strikes and 16 frames ending at apparent contact.</p>
<p>The receiver’s contact hip defines the origin. Time colors and pixel scales match. Body size and camera perspective are not normalized, so differences are descriptive and do not establish technique quality. All shinai lines are solid, including uncertain blur estimates; lengths are schematic.</p>
<p><a href="../average_strike/index.html">Person A average and variation</a> · <a href="../average_strike_second/index.html">Person B average and variation</a> · <a href="../strike_analysis_second/index.html">Person B five individual strikes</a></p>
<a href="two_average_strikes.svg">Overview SVG</a><img src="two_average_strikes.png" alt="Side by side mean strikes">
<h2>Matching instants: blue A, orange B</h2><a href="matched_mean_postures.svg">Frame comparison SVG</a><img src="matched_mean_postures.png" alt="Sixteen matching mean postures">
</html>''')
    from results_index import build_index
    build_index(ROOT/'outputs')
    print('Comparison:',OUT/'index.html','; max time mismatch:',delta,'ms')


if __name__=='__main__':main()
