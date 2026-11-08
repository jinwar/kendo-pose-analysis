"""Read presentation timestamps using FFmpeg, avoiding OpenCV's initial-PTS bug."""
import re
import subprocess
from bisect import bisect_left
import imageio_ffmpeg


def frame_range(times, start=0, duration=None):
    first=bisect_left(times,start)
    stop=len(times) if duration is None else bisect_left(times,start+duration)
    return first,stop


def timestamps(path):
    result=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner',
        '-i',str(path),'-map','0:v:0','-vf','showinfo=checksum=0','-an',
        '-f','null','-'],capture_output=True,text=True,check=True)
    times=[float(t) for t in re.findall(r'\bn:\s*\d+\s+pts:\s*\S+\s+pts_time:([-+\deE.]+)',result.stderr)]
    if not times or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('Missing or nonmonotonic video presentation timestamps')
    return times
