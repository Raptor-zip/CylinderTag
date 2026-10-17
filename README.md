# CylinderTag: An Accurate and Flexible Marker for Cylinder-Shape Objects Pose Estimation Based on Projective Invariants

<div align="center"> <img src="pose.bmp" width = 480 height = 300 /> </div>

## Table of Contents

- [Background](#background)
- [Requirements](#Requirements)
- [Usage](#usage)
	- [Generator](#generator)
  - [Recognizer](#Recognizer)
- [License](#License)

## Background

We propose a novel visual marker called CylinderTag, which is specifically designed for curved surfaces. CylinderTag is a cyclic marker crafted to be firmly attached to objects with cylinder-like shapes. It is encoded by leveraging the manifold assumption for surfaces with cylindrical properties and through the utilization of cross-ratio in projection invariant.

## Requirements

1. OpenCV 4 (test on 4.5.3)
2. Ceres 2 (test on 2.0)

## Usage

We provide a basic CylinderTag (12c2f) for use, if you need other kinds of markers (such as 15c3f, 18c4f, etc.) need to be generated automatically by the user using the generator. For the recognizer, a marker encoding file (.marker) and a 3D model file (.model) are required for proper use, and a camera parameter file (cameraParams.yaml) is required for positioning.

### Generator

We provide a simple MATLAB-based marker generator where the user can modify the type of marker, the number of markers generated, and draw the markers from within the generator.

### Recognizer

We have defined a CylinderTag class in the recognizer, which contains the functions detect, estimatePose, drawAxis, etc. You can find its usage and definition in the source file.

### Thin 3D-printed test marker

See [the PLA single-nozzle test strip](models/README.md) for a parametric ID 0
marker designed to wrap around an approximately 30 mm diameter cylinder,
STL generation, and manual white-to-black filament change instructions.

### Recorded-video detection and overlays

The headless detector processes full-resolution frames using local min/max
thresholds at three window sizes, convex contour candidates, gradient-based
edge line fitting, geometric column pairing, and continuous cross-ratio scores
across all dictionary IDs, phases, directions, and reflections. It requires
at least four supported columns and a margin over the runner-up hypothesis.
The core detector has no preference for ID 0. The overlay tool highlights ID 0
in green and other IDs in orange for this particular printed-strip experiment.

```bash
cmake -S . -B build/linux -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/linux -j 4
ctest --test-dir build/linux --output-on-failure
mkdir -p out/IMG_2422_improved
build/linux/cylindertag_video IMG_2422.mov CTag_2f12c.marker out/IMG_2422_improved/improved 1 5 1 0,62,87,131,200,250 1
python3 analyze_video_results.py out/IMG_2422_improved
ffmpeg -i out/IMG_2422_improved/improved.avi -i IMG_2422.mov -map 0:v:0 -map '1:a:0?' -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac -movflags +faststart -shortest out/IMG_2422_improved/IMG_2422_improved.mp4
```

Optional positional arguments after the output prefix are input scale (1),
base threshold tile width (5), gradient refinement enabled (1), comma-separated
snapshot frame indices (empty by default), and multiple threshold windows
(enabled by default; use 0 for a single window). The three tile widths are
`base`, `2*base+1`, and `4*base+1`. Input must be 8-bit grayscale for the C++
`detect()` API; the video tool converts decoded color frames automatically.

Outputs include MJPEG AVI, per-frame CSV and JSONL, `*.stages.jsonl` with
candidate codes and cross ratios, and selected intermediate PNGs (grayscale,
binary mask, quads, column pairs, candidate groups, and decoded results).
PNG/statistics diagnostics describe the strongest threshold pass; the final
result combines accepted spatial hypotheses from all passes. `selected_window`
records that pass. JSONL detections include `mirrored`, `decode_error`,
`decode_margin`, and `support_columns`. Corners in detection JSONL are expressed
at original frame resolution. Output is constant average input FPS; timestamps
are frame index / FPS rather than original VFR timestamps. Missing detections
are not filled by tracking. Camera calibration and a physical model are needed
separately for pose estimation; these overlays only evaluate ID detection.

On `IMG_2422.mov` (480 × 854, 266 frames), the original detector at commit
`00b147f` output ID 0 in 1/266 frames (0.38%) and other IDs in 55 frames.
The improved detector outputs ID 0 in **170/266 frames (63.91%)**, with no other
IDs in this clip. Development used this same clip, so this is not a held-out
performance estimate. The initial STL was vertically reflected relative to
the image dictionary; its generator is corrected, and the detector supports
already-printed reflected markers. No reprint is needed for this experiment.

The [LaTeX report](report/cylindertag_report.tex) documents the pipeline,
source bugs, references to ArUco/AprilTag, development comparisons, and remaining
failures. The local PDF is `report/cylindertag_report.pdf`; comparison video is
`out/IMG_2422_improved/IMG_2422_before_after.mp4`. Generated exports are ignored;
user-supplied input videos remain outside those ignore rules.

The decoder test covers 984 dictionary/phase/direction/reflection/missing-column
cases plus rejection of short, blank, and tiny inputs. Independent synthetic
image validation can be generated with:

```bash
python3 tests/generate_validation_video.py out/validation
build/linux/cylindertag_video out/validation/validation.avi CTag_2f12c.marker out/validation/improved 1 5 1
python3 tests/evaluate_validation.py out/validation/labels.json out/validation/improved.jsonl
```

The rendered-image check detects the correct ID in 59/164 positive images,
with one wrong-ID frame, and no detections in 36 negative images. Small/distant
markers remain a substantial limitation (3/82 correct in the smaller-size
condition). This is not a measured comparison against ArUco/AprilTag detectors.

Report figures are regenerated by `python3 report/build_figures.py`, which
expects the saved development runs in `out/`, followed by two LuaLaTeX passes
inside `report/`. Source commit `00b147f` can reconstruct the old detector.

## License

[MIT](LICENSE)

## Citation
If you utilize this code for any academic purposes, kindly reference the following work in your citations:

```bibtex
@article{wang2024cylindertag,
  title={CylinderTag: An Accurate and Flexible Marker for Cylinder-Shape Objects Pose Estimation Based on Projective Invariants},
  author={Wang, Shaoan and Zhu, Mingzhu and Hu, Yaoqing and Li, Dongyue and Yuan, Fusong and Yu, Junzhi},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2024},
  publisher={IEEE}
}
```
