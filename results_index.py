"""Refresh the local landing page from completed runs only."""
import html
import json
from pathlib import Path


def build_index(root):
    cards=[]
    for p in sorted(root.glob('*/summary.json'),key=lambda p:p.stat().st_mtime,reverse=True):
        if p.parent.name.startswith('smoke'):
            continue
        data=json.loads(p.read_text())
        name=html.escape(p.parent.name,quote=True)
        source=html.escape(Path(data['source']).name)
        count=data['processed_frames']
        cards.append(f'<article><a href="{name}/index.html"><img src="{name}/preview.jpg" alt="Pose comparison preview">'
                     f'<h2>{source}</h2></a><p>{count:,} frames · {count/data["fps"]:.1f} seconds · '
                     f'<a href="{name}/comparison.mp4">Open video</a></p></article>')
    strike_link = ('<article><h2><a href="strike_analysis/index.html">RTMPose: five strikes aligned to contact</a></h2><p>Posture-only figures, one row per strike, with every consecutive frame leading up to contact.</p></article>'
                   if (root/'strike_analysis/index.html').exists() else '')
    if (root/'average_strike/index.html').exists():
        strike_link += '<article><h2><a href="average_strike/index.html">Typical strike: average of five repetitions</a></h2><p>Contact-aligned mean poses, shinai orientation, and between-strike variation.</p></article>'
    for folder,label in [('strike_analysis_second','Video 2: five strikes'),('average_strike_second','Video 2: average strike'),('average_comparison','Compare the two people’s average strikes')]:
        if (root/folder/'index.html').exists():
            strike_link += f'<article><h2><a href="{folder}/index.html">{label}</a></h2></article>'
    text='''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Kendo video analysis</title>
<style>body{font:16px system-ui;background:#101923;color:#e8eff5;max-width:1200px;margin:48px auto;padding:0 24px}
p{color:#bccbd8;line-height:1.6}h1{font-size:36px}a{color:#78d6e6;text-decoration:none}h2{font-size:20px}
article{background:#1d2c3a;padding:18px;border-radius:14px;margin:24px 0}img{width:100%;border-radius:8px}</style>
<h1>Kendo video analysis</h1><p>YOLO26s Pose and RTMPose-m · local prototype<br>
Choose a completed comparison to review synchronized skeletons, track IDs, slow motion, and raw data.</p>'''+strike_link+''.join(cards)+'''
<p>Diagnostics are not measured accuracy. The models infer joints hidden by armor; review fast strikes and occlusions carefully.</p></html>'''
    (root/'index.html').write_text(text)


if __name__=='__main__':
    build_index(Path(__file__).resolve().parent/'outputs')
