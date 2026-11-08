"""Check alignment, timestamps, data shapes and encoded video for a completed run."""
import json
import sys
from pathlib import Path
import cv2
import numpy as np


def validate(folder):
    summary=json.loads((folder/'summary.json').read_text())
    previous=-1
    count=0
    with (folder/'poses.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            assert row['frame']==summary['start_frame']+count
            assert row['time_seconds']>=0 and row['time_seconds']>previous
            previous=row['time_seconds']
            assert set(row['models'])=={'yolo','rtmpose'}
            for pipeline in row['models'].values():
                ids=[]
                for person in pipeline['people']:
                    ids.append(person['track_id'])
                    assert np.shape(person['keypoints_xy'])==(17,2)
                    assert np.shape(person['keypoint_scores'])==(17,)
                    assert np.isfinite(person['keypoints_xy']).all()
                    assert np.isfinite(person['keypoint_scores']).all()
                assert len(set(ids))==len(ids)
            count+=1
    assert count==summary['processed_frames']
    cap=cv2.VideoCapture(str(folder/'comparison.mp4'))
    decoded=0
    while True:
        ok, frame=cap.read()
        if not ok:
            break
        assert frame.shape[1]==summary['settings']['panel_width']*2
        decoded+=1
    cap.release()
    assert decoded==count,(decoded,count)
    assert (folder/'index.html').exists()
    print(f'PASS {folder.name}: {count} aligned records, monotonic timestamps, valid 17-joint arrays, {decoded} decoded output frames')


if __name__=='__main__':
    for arg in sys.argv[1:]:
        validate(Path(arg))
