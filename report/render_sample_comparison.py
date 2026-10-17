"""Render the sample comparison from raw frames and measured per-frame JSON."""
import json
import subprocess
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'out/public_samples'
FIG = ROOT / 'report/generated'
FIG.mkdir(parents=True, exist_ok=True)
summary = json.loads((OUT / 'summary.json').read_text())
item = summary['videos'][0]
runs = item['runs']
rows = [[json.loads(x) for x in (OUT / ('test_' + r['label'] + '.jsonl')).read_text().splitlines()]
        for r in runs]
labels = ['Original (8fb42c9)', 'Bug fixes (#12)', 'Robust opt-in (#13)']
cap = cv2.VideoCapture(str(ROOT / item['path']))
fps = cap.get(cv2.CAP_PROP_FPS)
command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
           '-pix_fmt', 'bgr24', '-s', '1920x472', '-r', str(fps), '-i', '-',
           '-c:v', 'libx264', '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p',
           '-movflags', '+faststart', str(OUT / 'test_comparison.mp4')]
writer = subprocess.Popen(command, stdin=subprocess.PIPE)
index = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    panels = []
    for label, measured, run in zip(labels, rows, runs):
        assert measured[index]['frame'] == index
        image = cv2.resize(frame, (640, 400), interpolation=cv2.INTER_AREA)
        scale = np.array([640/frame.shape[1], 400/frame.shape[0]])
        ids = []
        for marker in measured[index]['markers']:
            ids.append(marker['id'])
            for column in marker['corners']:
                for start in [0, 4]:
                    points = np.round(np.array(column[start:start+4]) * scale).astype(np.int32)
                    cv2.polylines(image, [points], True, (255, 190, 20), 1, cv2.LINE_AA)
        panel = np.zeros((472, 640, 3), np.uint8)
        panel[72:] = image
        cv2.putText(panel, label, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1, cv2.LINE_AA)
        message = 'IDs: ' + ','.join(map(str, ids)) if ids else 'NO DETECTION'
        cv2.putText(panel, message + f" | {run['detected_frames']}/{run['frames']} frames",
                    (12, 46), cv2.FONT_HERSHEY_SIMPLEX, .5, (230,230,230) if ids else (70,70,255), 1, cv2.LINE_AA)
        cv2.putText(panel, f'Frame {index} | no ground-truth annotations', (12, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, (180,180,180), 1, cv2.LINE_AA)
        panels.append(panel)
    joined = np.hstack(panels)
    writer.stdin.write(joined.tobytes())
    if index in (0, 93):
        cv2.imwrite(str(OUT / f'comparison_frame_{index:04d}.png'), joined)
        # Crop the physical marker from the raw image, not an algorithm result.
        cv2.imwrite(str(FIG / f'public_sample_{index:04d}.png'), frame[285:790,850:1110])
    index += 1
cap.release()
writer.stdin.close()
assert writer.wait() == 0
assert index == runs[0]['frames'] == 259
fig, axes = plt.subplots(3, 1, figsize=(8, 2.5), sharex=True, constrained_layout=True)
for ax, measured, label in zip(axes, rows, labels):
    present = [bool(x['markers']) for x in measured]
    ax.scatter(range(len(present)), [1]*len(present),
               c=['#257bb0' if p else '#df5348' for p in present], marker='|', s=75)
    ax.set_yticks([])
    ax.set_ylim(.7, 1.3)
    ax.set_title(label, loc='left', fontsize=9)
axes[-1].set_xlabel('Frame (blue: detection; red: no detection). No ground-truth annotations.', fontsize=9)
fig.savefig(FIG / 'public_sample_timeline.png', dpi=180)
plt.close(fig)
print('Rendered', index, 'frames')
