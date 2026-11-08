"""Download public model weights into this project; never upload videos."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
import requests

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'models'
SOURCES = {
    'yolo26s-pose.pt': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt',
    'yolox_m.onnx': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip',
    'rtmpose_m.onnx': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip',
}

def main():
    MODELS.mkdir(exist_ok=True)
    manifest = {}
    for name, url in SOURCES.items():
        target = MODELS / name
        if not target.exists():
            print('Downloading', name, flush=True)
            temp = target.with_suffix('.download')
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with temp.open('wb') as out:
                    for chunk in response.iter_content(1024 * 1024):
                        out.write(chunk)
            if url.endswith('.zip'):
                with zipfile.ZipFile(temp) as archive:
                    members = [n for n in archive.namelist() if n.endswith('.onnx')]
                    if len(members) != 1:
                        raise ValueError(f'Expected one ONNX model: {members}')
                    with archive.open(members[0]) as src, target.open('wb') as out:
                        shutil.copyfileobj(src, out)
                temp.unlink()
            else:
                temp.replace(target)
        manifest[name] = {'url': url, 'bytes': target.stat().st_size,
                          'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    (MODELS / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Models ready:', MODELS)

if __name__ == '__main__':
    main()
