"""Compare detector executables against the same mirror-corrected dictionary.

The original and bugfix executables must expose the legacy headless video CLI;
the improved executable is this repository's cylindertag_video. No ROI, ID-only
dictionary, video reflection, or tracking is applied.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mirrored_dictionary(source, output):
    tokens = list(map(int, source.read_text().split()))
    count, columns, features = tokens[:3]
    codes = tokens[3:]
    if len(codes) != count * columns or not all(0 <= c < 64 for c in codes):
        raise ValueError('Invalid marker dictionary')
    rows = [codes[i*columns:(i+1)*columns] for i in range(count)]
    output.write_text(f'{count} {columns} {features}\n' + '\n'.join(
        ' '.join(str(63-code) for code in row) for row in rows) + '\n')
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--improved', type=Path, required=True)
    parser.add_argument('--bugfix', type=Path)
    parser.add_argument('--video', type=Path, default=Path('IMG_2422.mov'))
    parser.add_argument('--dictionary', type=Path, default=Path('CTag_2f12c.marker'))
    parser.add_argument('--output', type=Path, default=Path('out/fair_comparison'))
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    video, dictionary = args.video.resolve(), args.dictionary.resolve()
    corrected = folder/'mirrored.marker'
    count = mirrored_dictionary(dictionary, corrected)
    probe = subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=nb_read_frames','-of','json',str(video)], text=True)
    frames = int(json.loads(probe)['streams'][0]['nb_read_frames'])
    manifest = dict(video_sha256=sha256(video), dictionary_sha256=sha256(dictionary),
        mirrored_dictionary_sha256=sha256(corrected), dictionary_ids=count,
        dictionary_transform='c -> 63-c for every entry; keep all IDs',
        video_modified=False, input_scale=1, threshold_window=5,
        subpixel_refinement=True, expected_frames=frames, runs=[])
    runs = [('original_verified' if i == 0 else f'original_repeat_{i}',args.original,False)
            for i in range(args.repeats)]
    if args.bugfix:
        runs.append(('bugfix_only',args.bugfix,False))
    runs.append(('improved_mirror',args.improved,True))
    for name, binary, improved in runs:
        binary = binary.resolve()
        prefix = folder/name
        command = [str(binary),str(video),str(corrected),str(prefix),'1','5','1']
        if improved:
            command += ['87','1']
        with prefix.with_suffix('.log').open('w') as log:
            subprocess.run(command, stdout=log, check=True)
        rows = list(csv.DictReader(prefix.with_suffix('.csv').open()))
        assert len(rows) == frames and all(int(row['frame']) == i for i,row in enumerate(rows))
        record = dict(run=name, executable_sha256=sha256(binary), command=command,
            frames=len(rows), id0_frames=sum(int(row['id0_detected']) for row in rows),
            other_id_frames=sum(any(x != '0' for x in row['ids'].split(';') if x) for row in rows))
        record['id0_percent'] = 100*record['id0_frames']/frames
        manifest['runs'].append(record)
        print(f"{name}: ID0={record['id0_frames']}/{frames} ({record['id0_percent']:.2f}%), other={record['other_id_frames']}")
    (folder/'measurement_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
