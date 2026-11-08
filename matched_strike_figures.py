"""Export six matching 4x4 figures: five original strike pairs and their means."""
import json
import zipfile
from pathlib import Path
from compare_individual_strikes import origin, draw_frame, COLORS
from average_strikes import draw_body, draw_shinai, setup_axis
from plot_strikes import plt, save

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/matched_strike_figures'


def main():
    sources=['outputs/strike_analysis/aligned_poses.json','outputs/strike_analysis_second/aligned_poses.json',
             'outputs/average_strike/average_strike.json','outputs/average_strike_second/average_strike.json']
    a,b,mean_a,mean_b=[json.loads((ROOT/p).read_text()) for p in sources]
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=[];cards=[]
    for k in range(6):
        average=k==5
        pair=[mean_a['frames'],mean_b['frames']] if average else [a['sequences'][k],b['sequences'][k]]
        if any([r['offset_frames'] for r in s]!=list(range(-15,1)) for s in pair):
            raise ValueError('Each figure requires 16 frames ending at contact')
        anchors=None if average else [origin(s) for s in pair]
        fig,axes=plt.subplots(4,4,figsize=(17,17))
        fig.subplots_adjust(left=.025,right=.99,top=.9,bottom=.075,hspace=.25,wspace=.03)
        for i,ax in enumerate(axes.flat):
            for j,(sequence,color) in enumerate(zip(pair,COLORS)):
                if average:
                    draw_body(ax,sequence[i]['joints'],color,width=1.5,alpha=.85)
                    draw_shinai(ax,sequence[i]['shinai'],color)
                else:draw_frame(ax,sequence[i],anchors[j],color,[a,b][j]['threshold'])
            setup_axis(ax)
            ax.set_title(f"A {pair[0][i]['relative_time_ms']:.0f} / B {pair[1][i]['relative_time_ms']:.0f} ms"+(' · contact' if i==15 else ''),fontsize=11)
        title='Average strike · mean postures frame by frame' if average else f'Strike {k+1} · original postures frame by frame'
        fig.suptitle(title,fontsize=22,fontweight='bold',x=.025,ha='left')
        fig.text(.025,.94,'Blue: Person A / video 1    Orange: Person B / video 2    ·    Read left to right, then down',fontsize=13)
        if average:
            note='Confidence-weighted average of five strikes per person, aligned to each receiver’s contact hip. No time warping.\nSolid shinai lines show mean orientation with schematic length, not measured tips. No body-size or perspective correction.'
        else:
            note='Each strike is translated once to its receiver’s contact hip. No averaging, interpolation or body-size correction.\nSolid shinai segments include uncertain blur estimates; endpoints are not verified tips. Manual contact ≈ ±1 frame.'
        fig.text(.025,.025,note,fontsize=11,color='#52606d')
        name='06_average_strike' if average else f'{k+1:02d}_strike_{k+1}'
        save(fig,OUT/name)
        manifest.append({'figure':k+1,'name':name,'title':title,'type':'average' if average else 'individual',
                         'offset_frames':list(range(-15,1)),
                         'times_ms':[[r['relative_time_ms'] for r in s] for s in pair]})
        cards.append(f'<section><h2>{k+1}. {title}</h2><p><a href="{name}.png">PNG</a> · <a href="{name}.svg">Vector SVG</a></p><img src="{name}.png" alt="{title}" loading="lazy"></section>')
    (OUT/'manifest.json').write_text(json.dumps({'sources':sources,'figures':manifest},indent=2))
    with zipfile.ZipFile(OUT/'six_figures.zip','w',zipfile.ZIP_DEFLATED) as bundle:
        for item in manifest:
            for suffix in ['png','svg']:
                p=OUT/(item['name']+'.'+suffix);bundle.write(p,p.name)
        bundle.write(OUT/'manifest.json','manifest.json')
    (OUT/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Six matched strike figures</title>
<style>body{font:16px system-ui;max-width:1450px;margin:32px auto;padding:20px;color:#243344}img{width:100%}p{line-height:1.6}section{margin:48px 0}a{color:#1479b8}</style>
<h1>Five strikes and the average — six matched figures</h1>
<p>Blue: Person A, earlier recording (12:02:41). Orange: Person B, later recording (12:03:35). Each figure has 16 panels from approximately −0.5 seconds to contact. Strike pairs follow recording order. All figures share axes, colors and layout.</p>
<p>The first five figures show original observations; the sixth compares each person’s confidence-weighted average of five strikes. All shinai lines are solid. Individual segments show detected shaft support; the average uses schematic orientation guides. Body size and camera perspective are not normalized; blurred shinai estimates and occluded joints remain uncertain.</p>
<p><a href="six_figures.zip">Download all six figures (PNG + SVG)</a> · <a href="manifest.json">Sources and timing</a></p>'''+''.join(cards)+'</html>')
    from results_index import build_index
    build_index(ROOT/'outputs')
    print('Created six 4×4 figures:',OUT/'index.html')


if __name__=='__main__':main()
