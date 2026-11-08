"""Run two independent pose pipelines on identical decoded video frames."""
import argparse
import csv
import html
import json
import os
from pathlib import Path
import platform
import time
from collections import defaultdict
from importlib.metadata import version

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.cache' / 'ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')

import cv2
import numpy as np
import imageio_ffmpeg
import torch
from ultralytics import YOLO
from rtmlib import YOLOX, RTMPose
from tracking import Tracker
from results_index import build_index
from video_timing import timestamps, frame_range

JOINTS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
          'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
          'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
          'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
EDGES = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),
         (11,13),(13,15),(12,14),(14,16),(0,1),(0,2),(1,3),(2,4)]
COLORS = [(90,220,255),(255,190,70),(180,120,255),(120,240,140),(200,160,80)]


class Models:
    def __init__(self, device, size, det_threshold):
        self.device, self.size, self.det_threshold = device, size, det_threshold
        self.yolo = YOLO(str(ROOT / 'models/yolo26s-pose.pt'))
        self.detector = YOLOX(str(ROOT / 'models/yolox_m.onnx'),
                             model_input_size=(640, 640), score_thr=det_threshold,
                             backend='onnxruntime', device='cpu')
        self.pose = RTMPose(str(ROOT / 'models/rtmpose_m.onnx'),
                            model_input_size=(192, 256),
                            backend='onnxruntime', device='cpu')

    def infer(self, name, frame):
        if name == 'yolo':
            result = self.yolo.predict(frame, imgsz=self.size, device=self.device,
                                       conf=self.det_threshold, verbose=False)[0].cpu()
            boxes = result.boxes.xyxy.numpy()
            if not len(boxes):
                return boxes, np.empty((0,17,2)), np.empty((0,17)), []
            return (boxes, result.keypoints.xy.numpy(), result.keypoints.conf.numpy(),
                    result.boxes.conf.numpy().tolist())
        boxes = np.asarray(self.detector(frame), dtype=np.float32).reshape(-1,4)
        # RTMLib defaults to full-image pose when bboxes is empty: explicitly avoid that.
        if not len(boxes):
            return boxes, np.empty((0,17,2)), np.empty((0,17)), []
        points, scores = self.pose(frame, bboxes=boxes)
        # RTMLib's YOLOX interface discards detection confidence; do not invent it.
        return boxes, points, scores, [None] * len(boxes)


def draw(frame, boxes, points, scores, ids, name, source_time, frame_index, threshold, width):
    scale = width / frame.shape[1]
    height = int(round(frame.shape[0] * scale / 2)) * 2
    canvas = cv2.resize(frame, (width, height))
    for box, pose, conf, identity in zip(boxes, points, scores, ids):
        color = COLORS[(identity - 1) % len(COLORS)]
        x1,y1,x2,y2 = np.rint(box * [scale, height/frame.shape[0], scale, height/frame.shape[0]]).astype(int)
        cv2.rectangle(canvas,(x1,y1),(x2,y2),color,1)
        cv2.putText(canvas,f'ID {identity}',(max(0,x1),max(18,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
        xy = np.rint(pose * [scale, height/frame.shape[0]]).astype(int)
        valid = np.isfinite(conf) & (conf >= threshold)
        for a,b in EDGES:
            if valid[a] and valid[b]:
                cv2.line(canvas,tuple(xy[a]),tuple(xy[b]),color,2,cv2.LINE_AA)
        for i,p in enumerate(xy):
            if valid[i]:
                cv2.circle(canvas,tuple(p),4 if i in (9,10) else 3,color,-1,cv2.LINE_AA)
    header = np.full((64,width,3),(28,24,20),np.uint8)
    title = 'YOLO26s Pose' if name == 'yolo' else 'YOLOX-m + RTMPose-m'
    cv2.putText(header,title,(16,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),1,cv2.LINE_AA)
    cv2.putText(header,f'{source_time:.3f}s | frame {frame_index} | {len(ids)} people | IDs local to panel',
                (16,49),cv2.FONT_HERSHEY_SIMPLEX,.42,(190,200,210),1,cv2.LINE_AA)
    return np.vstack([header,canvas])


def report(out, summary):
    rows = ''
    for name, m in summary['models'].items():
        rows += ('<tr><td>'+name+'</td><td>'+str(m['unique_track_ids'])+'</td><td>'
                 +str(m['frames_without_detection'])+'</td><td>'
                 +f"{m['mean_people_per_frame']:.2f}</td><td>{m['visible_body_fraction']:.1%}</td>"
                 +f"<td>{m['visible_wrist_fraction']:.1%}</td><td>{m['inference_fps']:.1f}</td></tr>")
    title = html.escape(Path(summary['source']).name)
    document = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Kendo pose comparison</title>
<style>body{font:16px system-ui;background:#101923;color:#e8eff5;max-width:1450px;margin:32px auto;padding:0 24px}
h1{font-size:30px}p{color:#bccbd8;line-height:1.6}video{width:100%;background:#000;border-radius:10px}
button,select{padding:9px 14px;margin:6px;background:#263a4b;color:white;border:1px solid #506477;border-radius:6px}
table{width:100%;border-collapse:collapse;font-size:14px;margin:24px 0}td,th{text-align:left;padding:12px;border-bottom:1px solid #334556}
a{color:#78d6e6}.note{padding:16px;background:#1d2c3a;border-radius:10px}code{color:#8ce0df}</style>
<h1>Kendo · Pose comparison</h1><p>__TITLE__ · identical source frames · independent IDs · shared motion/IoU tracker</p>
<video id="v" controls preload="metadata" src="comparison.mp4"></video>
<div><button onclick="step(-1)">← Frame</button><button onclick="step(1)">Frame →</button>
<label>Playback <select onchange="v.playbackRate=Number(this.value)"><option value="0.25">¼ speed</option><option value="0.5">½ speed</option><option value="1" selected>Normal</option></select></label>
<span id="stamp"></span></div>
<table><tr><th>Pipeline</th><th>Unique IDs</th><th>Empty frames</th><th>Mean people</th><th>Body ≥ threshold</th><th>Wrists ≥ threshold</th><th>Inference fps</th></tr>__ROWS__</table>
<p class="note">These are diagnostics, not accuracy scores. Confidence scales differ between models.
Coverage is conditional on detected people (shoulders through ankles; wrists separately), not ground truth.
More IDs may reflect new people, mirrors, false positives, or fragmentation; they are not measured identity switches.
The baseline tracker has no appearance re-identification. IDs and colors need not match between panels.
No interpolation or temporal smoothing is applied to keypoints. Source audio is omitted.</p>
<p>Use the frame controls to inspect wrist placement, hidden elbows and knees, missed detections, and IDs during turns.
Shinai tracking and strike classification are not implemented.</p>
<p><a href="summary.json">Run settings and diagnostics</a> · <a href="frames.csv">Per-frame diagnostics</a> ·
<a href="poses.jsonl">Raw landmarks and track IDs</a> · <a href="preview.jpg">Preview image</a></p>
<script>const v=document.getElementById('v'),fps=__FPS__,start=__START__;
function step(d){v.pause();let n=Math.round(v.currentTime*fps)+d;v.currentTime=Math.max(0,Math.min(v.duration-1/fps,n/fps));}
v.addEventListener('timeupdate',()=>{document.getElementById('stamp').textContent='Source ≈ '+(start+v.currentTime).toFixed(3)+' s';});
document.addEventListener('keydown',e=>{if(e.target.tagName==='SELECT')return;if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();step(e.key==='ArrowLeft'?-1:1);}});</script></html>'''
    document = document.replace('__TITLE__',title).replace('__ROWS__',rows)
    document = document.replace('__FPS__',str(summary['fps'])).replace('__START__',str(summary['start_frame']/summary['fps']))
    (out/'index.html').write_text(document)


def run(args):
    source = args.video.resolve()
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {source}')
    fps, reported_total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_times = timestamps(source)
    total = len(source_times)
    if fps <= 0 or total <= 0:
        raise ValueError('Video has invalid frame rate/frame count')
    first, stop = frame_range(source_times,args.start,args.duration)
    if first >= stop:
        raise ValueError('Requested range contains no frames')
    out = args.output.resolve() if args.output else ROOT/'outputs'/f'{source.stem}_{time.strftime("%Y%m%d_%H%M%S")}'
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f'Output folder is not empty; use a new folder: {out}')
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    models = Models(args.device, args.imgsz, args.det_threshold)
    ok, warm = cap.read()
    if not ok:
        raise ValueError('Cannot decode first frame')
    names = ['yolo','rtmpose']
    print('Warming up both models...', flush=True)
    for name in names:
        models.infer(name,warm)
    cap.release()
    cap = cv2.VideoCapture(str(source))
    # Exact source indices: decode/skip from the beginning instead of keyframe seeking.
    for _ in range(first):
        if not cap.grab():
            raise ValueError('Cannot skip to requested starting frame')
    trackers = {n:Tracker(max_missed=round(fps*.5)) for n in names}
    stats = {n:defaultdict(float) for n in names}
    seen = {n:set() for n in names}
    width = args.panel_width
    height = int(round(warm.shape[0]/warm.shape[1]*width/2))*2 + 64
    encoder = imageio_ffmpeg.write_frames(str(out/'comparison.mp4'),(width*2,height),
                fps=fps,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',
                macro_block_size=2,ffmpeg_log_level='error',
                output_params=['-crf','20','-preset','fast','-movflags','+faststart'])
    encoder.send(None)
    wall_start = time.perf_counter()
    count = 0
    try:
        with (out/'poses.jsonl').open('w') as raw, (out/'frames.csv').open('w',newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['frame','time_seconds','pipeline','people','track_ids','inference_ms','body_visible','body_total','wrists_visible','wrists_total'])
            for frame_index in range(first,stop):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f'Decode failed at frame {frame_index}; partial outputs saved')
                decoder_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                timestamp = source_times[frame_index]
                record = {'frame':frame_index,'time_seconds':timestamp,
                          'decoder_time_seconds':decoder_timestamp,'models':{}}
                panels = {}
                # Alternate execution order to avoid always giving one model the first slot.
                for name in (names if count%2==0 else names[::-1]):
                    tick=time.perf_counter()
                    boxes,points,scores,det_scores=models.infer(name,frame)
                    if args.device == 'mps' and name == 'yolo':
                        torch.mps.synchronize()
                    elapsed=time.perf_counter()-tick
                    ids=trackers[name].update(boxes)
                    valid=(scores>=args.keypoint_threshold)&np.isfinite(scores)
                    body=int(valid[:,5:17].sum()); wrists=int(valid[:,[9,10]].sum())
                    stat=stats[name]
                    stat['seconds']+=elapsed; stat['detections']+=len(boxes)
                    stat['empty']+=int(len(boxes)==0); stat['body']+=body; stat['wrists']+=wrists
                    seen[name].update(ids)
                    people=[]
                    for i, identity in enumerate(ids):
                        people.append({'track_id':identity,'bbox_xyxy':boxes[i].tolist(),
                            'detection_confidence':det_scores[i], 'keypoints_xy':points[i].tolist(),
                            'keypoint_scores':scores[i].tolist()})
                    record['models'][name]={'inference_ms':elapsed*1000,'people':people}
                    writer.writerow([frame_index,timestamp,name,len(boxes),' '.join(map(str,ids)),
                                     elapsed*1000,body,len(boxes)*12,wrists,len(boxes)*2])
                    panels[name]=draw(frame,boxes,points,scores,ids,name,timestamp,frame_index,args.keypoint_threshold,width)
                raw.write(json.dumps(record,allow_nan=False)+'\n')
                combined=np.hstack([panels[n] for n in names])
                encoder.send(np.ascontiguousarray(cv2.cvtColor(combined,cv2.COLOR_BGR2RGB)))
                if count==0:
                    cv2.imwrite(str(out/'preview.jpg'),combined)
                count+=1
                if count%30==0:
                    print(f'{count}/{stop-first} frames | {time.perf_counter()-wall_start:.1f}s elapsed',flush=True)
    finally:
        cap.release()
        encoder.close()
    summary={'source':str(source),'source_size':[warm.shape[1],warm.shape[0]],
        'source_frames':total,'container_reported_frames':reported_total,
        'fps':fps,'timestamp_basis':'FFmpeg presentation timestamps',
        'start_frame':first,'end_frame_exclusive':stop,
        'processed_frames':count,'keypoint_names':JOINTS,'coordinate_system':'original video pixels',
        'elapsed_seconds':time.perf_counter()-wall_start,'platform':platform.platform(),
        'settings':vars(args).copy(),'models':{},
        'versions':{p:version(p) for p in ['ultralytics','rtmlib','torch','onnxruntime','numpy','opencv-python']},
        'model_manifest':json.loads((ROOT/'models/manifest.json').read_text()),
        'method':'Same decoded frames; independent native person detectors; shared motion/IoU Hungarian tracker. RTMPose uses YOLOX-m 640 and 192x256 person crops. No keypoint smoothing. RTMLib YOLOX includes an internal additional score filter at its NMS threshold (0.45). Detection/confidence settings are not calibrated across models.',
        'limitations':['No ground truth: coverage is not accuracy; unique IDs are not ID switches.',
                      'RTMPose uses ONNX Runtime CPU; YOLO backend is recorded in settings.',
                      'Inference fps excludes warmup, decoding, tracking, drawing and encoding.',
                      'Video output uses nominal constant fps; raw data retains presentation timestamps.',
                      'No shinai detection, strike classification, or guaranteed identity through occlusion.']}
    summary['settings']={k:str(v) if isinstance(v,Path) else v for k,v in summary['settings'].items()}
    for name in names:
        s=stats[name]; d=s['detections']
        summary['models'][name]={'unique_track_ids':len(seen[name]),'frames_without_detection':int(s['empty']),
            'mean_people_per_frame':d/count,'visible_body_fraction':s['body']/(12*d) if d else 0,
            'visible_wrist_fraction':s['wrists']/(2*d) if d else 0,'inference_fps':count/s['seconds']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    report(out,summary)
    build_index(ROOT/'outputs')
    print('Finished:',out/'index.html',flush=True)
    print(json.dumps(summary['models'],indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video',type=Path)
    parser.add_argument('--start',type=float,default=0)
    parser.add_argument('--duration',type=float,help='Seconds; omit to process through the end')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu',help='YOLO backend; RTMPose uses CPU')
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--det-threshold',type=float,default=.5)
    parser.add_argument('--keypoint-threshold',type=float,default=.3)
    parser.add_argument('--panel-width',type=int,default=960)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.start<0 or (args.duration is not None and args.duration<=0):
        parser.error('start must be nonnegative and duration must be positive')
    if args.panel_width<320 or args.panel_width%2 or args.imgsz<32 or args.threads<1:
        parser.error('Use even panel-width >=320, imgsz >=32, threads >=1')
    if not 0<args.det_threshold<1 or not 0<args.keypoint_threshold<1:
        parser.error('Confidence thresholds must be between 0 and 1')
    run(args)

if __name__=='__main__':
    main()
