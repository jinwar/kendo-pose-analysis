"""Re-render a completed run from saved poses; no model inference required.

Refreshes presentation timestamps from FFmpeg as well. Useful when updating
overlay style without paying the cost of another inference run.
"""
import csv
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import imageio_ffmpeg
from compare import draw, report, ROOT
from results_index import build_index
from video_timing import timestamps


def render(folder):
    summary=json.loads((folder/'summary.json').read_text())
    times=timestamps(summary['source'])
    cap=cv2.VideoCapture(summary['source'])
    for _ in range(summary['start_frame']):
        if not cap.grab():raise ValueError('Cannot skip source frame')
    width=summary['settings']['panel_width']
    sw,sh=summary['source_size']
    height=int(round(sh/sw*width/2))*2+64
    encoder=imageio_ffmpeg.write_frames(str(folder/'comparison_new.mp4'),(width*2,height),
        fps=summary['fps'],codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',
        macro_block_size=2,ffmpeg_log_level='error',
        output_params=['-crf','20','-preset','fast','-movflags','+faststart'])
    encoder.send(None)
    try:
        with (folder/'poses.jsonl').open() as src,(folder/'poses_new.jsonl').open('w') as dst:
            for i,line in enumerate(src):
                row=json.loads(line)
                ok,frame=cap.read()
                if not ok:raise ValueError('Source decode failed')
                row['time_seconds']=times[row['frame']]
                dst.write(json.dumps(row,allow_nan=False)+'\n')
                panels=[]
                for name in ['yolo','rtmpose']:
                    people=row['models'][name]['people']
                    panels.append(draw(frame,np.array([p['bbox_xyxy'] for p in people]),
                        np.array([p['keypoints_xy'] for p in people]),
                        np.array([p['keypoint_scores'] for p in people]),
                        [p['track_id'] for p in people],name,row['time_seconds'],row['frame'],
                        summary['settings']['keypoint_threshold'],width))
                combined=np.hstack(panels)
                encoder.send(np.ascontiguousarray(cv2.cvtColor(combined,cv2.COLOR_BGR2RGB)))
                if i==0:cv2.imwrite(str(folder/'preview.jpg'),combined)
                if row['frame'] in [60,150,300,450,600]:
                    cv2.imwrite(str(folder/f'review_frame_{row["frame"]:06d}.jpg'),combined)
    finally:
        cap.release();encoder.close()
    with (folder/'frames.csv').open() as src,(folder/'frames_new.csv').open('w',newline='') as dst:
        reader=csv.DictReader(src);writer=csv.DictWriter(dst,fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            row['time_seconds']=times[int(row['frame'])];writer.writerow(row)
    for name in ['comparison.mp4','poses.jsonl','frames.csv']:
        path=Path(name)
        (folder/(path.stem+'_new'+path.suffix)).replace(folder/path)
    summary.pop('decoder_timestamp_origin',None)
    summary['timestamp_basis']='FFmpeg presentation timestamps'
    summary['limitations']=[s.replace('raw data retains decoder timestamps','raw data retains presentation timestamps') for s in summary['limitations']]
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    report(folder,summary)
    build_index(ROOT/'outputs')
    print('Re-rendered:',folder,flush=True)


if __name__=='__main__':
    for arg in sys.argv[1:]:render(Path(arg))
