"""Build report figures from measured detector outputs (run at repository root)."""
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'report' / 'generated'
DEST.mkdir(parents=True, exist_ok=True)


def records(folder, name):
    p = ROOT / 'out' / folder / name
    rows = list(csv.DictReader(p.with_suffix('.csv').open()))
    detections = [json.loads(line) for line in p.with_suffix('.jsonl').open()]
    assert len(rows) == len(detections)
    return rows, detections


def metrics(folder, name):
    rows, dets = records(folder, name)
    ids = [{m['id'] for m in r['markers']} for r in dets]
    return dict(name=name, frames=len(rows), id0=sum(0 in x for x in ids),
                wrong=sum(bool(x-{0}) for x in ids))


runs = [('IMG_2422', 'baseline'), ('IMG_2422_improved', 'fullres'),
        ('IMG_2422_improved', 'mirror_probe'), ('IMG_2422_improved', 'softdecode'),
        ('IMG_2422_improved', 'gradient'), ('IMG_2422_improved', 'localcontrast'),
        ('IMG_2422_improved', 'improved')]
result = {'actual': [metrics(*x) for x in runs]}
labels = json.loads((ROOT/'out/validation/labels.json').read_text())
result['synthetic'] = []
for name in ['legacy', 'improved']:
    _, dets = records('validation', name)
    assert len(dets) == len(labels) == 200
    counts = dict(name=name, positives=164, correct=0, wrong=0, negatives=36, negative_fp=0,
                  near_correct=0, far_correct=0, mirror_errors=0)
    for label, row in zip(labels, dets):
        ids = {m['id'] for m in row['markers']}
        if label['id'] is None:
            counts['negative_fp'] += bool(ids)
        else:
            hit = label['id'] in ids
            counts['correct'] += hit
            counts['wrong'] += bool(ids-{label['id']})
            counts['near_correct' if label['variant']<2 else 'far_correct'] += hit
            counts['mirror_errors'] += any(m['id']==label['id'] and 'mirrored' in m and m['mirrored']!=label['mirrored'] for m in row['markers'])
    result['synthetic'].append(counts)
result['decoder_cases'] = 984
(DEST/'results.json').write_text(json.dumps(result, indent=2)+'\n')

# Actual intermediate buffers from frame 87, cropped identically to show bars.
for stage in ['01_gray','02_binary','03_quads','04_pairs','05_groups','06_decoded']:
    image=cv2.imread(str(ROOT/'out/IMG_2422_improved'/f'improved_frame_000087_{stage}.png'))
    assert image is not None
    crop=image[225:495,175:410]
    cv2.imwrite(str(DEST/f'pipeline_{stage}.png'), crop)

# Identical frame: old and improved overlays.
for folder,name,out in [('IMG_2422','baseline','before'),('IMG_2422_improved','improved','after')]:
    cap=cv2.VideoCapture(str(ROOT/'out'/folder/(name+'.avi')))
    cap.set(cv2.CAP_PROP_POS_FRAMES,87)
    ok,img=cap.read();cap.release();assert ok
    cv2.imwrite(str(DEST/(out+'.png')),img[225:495,160:460])

# A real failure near t=4.53 s, never replaced by a tracked estimate.
cap=cv2.VideoCapture(str(ROOT/'out/IMG_2422_improved/improved.avi'))
cap.set(cv2.CAP_PROP_POS_FRAMES,131);ok,img=cap.read();cap.release();assert ok
cv2.imwrite(str(DEST/'failure.png'), img)
cv2.imwrite(str(DEST/'failure_crop.png'), img[220:500,100:380])

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,axes=plt.subplots(2,1,figsize=(9,3.6),sharex=True,constrained_layout=True)
for ax,(folder,name),title in zip(axes,[('IMG_2422','baseline'),('IMG_2422_improved','improved')],['Original detector','Improved detector']):
    rows,dets=records(folder,name)
    times=[float(r['time_seconds']) for r in rows]
    good=[int(any(m['id']==0 for m in r['markers'])) for r in dets]
    bad=[int(any(m['id']!=0 for m in r['markers'])) for r in dets]
    ax.vlines([t for t,b in zip(times,bad) if b],-.95,-.1,color='#d46a28',linewidth=1.5,label='Other ID')
    ax.vlines([t for t,g in zip(times,good) if g],.1,.95,color='#168755',linewidth=1.5,label='ID 0')
    ax.set_ylim(-1.05,1.05);ax.set_yticks([-.5,.5],['Other ID','ID 0']);ax.set_title(title,loc='left',fontsize=10)
    ax.grid(axis='x',alpha=.2)
axes[-1].set_xlabel('Time (s), normalized to average FPS')
fig.savefig(DEST/'timeline.png',dpi=180);plt.close(fig)

# Stage retention: different pixel resolutions, so counts are diagnostic only.
fig,ax=plt.subplots(figsize=(7.7,2.6),constrained_layout=True)
keys=['quads','features','valid_columns'];titles=['Quadrilaterals','Paired columns','Valid edge geometry']
x=np.arange(len(keys));width=.32
for offset,name,label,color in [(-width/2,'instrumented','Original (half resolution)','#999999'),(width/2,'improved','Improved (selected window)','#168755')]:
    rows,_=records('IMG_2422_improved',name)
    values=[np.mean([int(r[k]) for r in rows]) for k in keys]
    bars=ax.bar(x+offset,values,width,label=label,color=color)
    ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
ax.set_xticks(x,titles);ax.set_ylabel('Mean count / frame');ax.legend(fontsize=8,frameon=False,loc='upper left')
ax.set_ylim(0,55);fig.savefig(DEST/'stage_counts.png',dpi=180);plt.close(fig)

# Independent synthetic validation example, not an image made by the detector.
cap=cv2.VideoCapture(str(ROOT/'out/validation/validation.avi'))
cap.set(cv2.CAP_PROP_POS_FRAMES,0);ok,img=cap.read();cap.release();assert ok
cv2.imwrite(str(DEST/'synthetic.png'),img)

# Faithful old-vs-corrected flat print pattern, in image coordinates.
# Avoid importing CAD dependencies: load only the pure geometry functions via AST.
import ast, math
source=ast.parse((ROOT/'models/cylindertag_d30.py').read_text())
names={'marker_codes','boundary','image_polygons'}
funcs=ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
namespace={'pi':math.pi,'sqrt':math.sqrt,'Path':Path,'DICTIONARY':ROOT/'CTag_2f12c.marker','MARKER_ID':0,
           'STRIP_LENGTH':math.pi*30.2,'STRIP_HEIGHT':44.,'AXIAL_MARGIN':2.,'PATTERN_HEIGHT':40.,'WHITE_RATIO':.2}
exec(compile(funcs,'model_functions','exec'),namespace)
reference=np.full((440,950,3),255,np.uint8)
for polygon in namespace['image_polygons']():
    cv2.fillPoly(reference,[np.round(np.array(polygon)*[950/namespace['STRIP_LENGTH'],10]).astype(np.int32)],(0,0,0))
cv2.imwrite(str(DEST/'pattern_corrected.png'),reference)
cv2.imwrite(str(DEST/'pattern_legacy_mirrored.png'),cv2.flip(reference,0))

# TeX numeric macros come only from the measured summaries.
before=result['actual'][0];after=result['actual'][-1]
macros={'BeforeCount':before['id0'],'AfterCount':after['id0'],'BeforeWrong':before['wrong'],'AfterWrong':after['wrong'],
        'FrameCount':after['frames'],'BeforeRate':f"{100*before['id0']/before['frames']:.2f}",
        'AfterRate':f"{100*after['id0']/after['frames']:.2f}",
        'SyntheticBefore':result['synthetic'][0]['correct'],'SyntheticAfter':result['synthetic'][1]['correct'],
        'SyntheticWrongBefore':result['synthetic'][0]['wrong'],'SyntheticWrongAfter':result['synthetic'][1]['wrong']}
(DEST/'metrics.tex').write_text('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k,v in macros.items())+'\n')
print(json.dumps(result,indent=2))
