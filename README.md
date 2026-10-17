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

The headless video tool decodes the full marker dictionary, writes per-frame
IDs and corners, and draws decoded quadrilateral outlines. Green indicates
the expected ID 0; orange indicates another decoded ID. Missed frames are
not filled using tracking. This evaluates detection, without pose estimation
or use of the supplied camera/model calibration.

```bash
cmake -S . -B build/linux
cmake --build build/linux --target cylindertag_video -j 4
mkdir -p out/IMG_2422
# Arguments after prefix: input scale, threshold window, subpixel refinement (0/1).
build/linux/cylindertag_video IMG_2422.mov CTag_2f12c.marker out/IMG_2422/baseline 1 5 1
build/linux/cylindertag_video IMG_2422.mov CTag_2f12c.marker out/IMG_2422/scale1_plain 1 5 0
python3 analyze_video_results.py out/IMG_2422
ffmpeg -i out/IMG_2422/scale1_plain.avi -i IMG_2422.mov -map 0:v:0 -map '1:a:0?' -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac -movflags +faststart -shortest out/IMG_2422/IMG_2422_overlay.mp4
```

Outputs are MJPEG AVI, CSV (IDs and detected-column counts), and JSONL
(decoded column indices and full-resolution corner coordinates). Each run
processes every decoded frame. The output uses constant average input FPS;
displayed timestamps are frame index / FPS rather than original VFR timestamps.
The input video is a manually supplied asset; generated analyses live in ignored `out/`.

For `IMG_2422.mov` (480 × 854, 266 frames, about 9.2 seconds), the standard
settings output ID 0 in 1/266 frames (0.38%), with other IDs in 55 frames.
Without subpixel refinement, ID 0 appears in frame 62 (about 2.15 s) with
four decoded columns whose outlines visually align with the printed marker;
other IDs appear in 24 frames. Seven scale/threshold/refinement configurations
all output ID 0 in only one frame. This is **not stable recognition**, and this
clip does not establish physical position or angle accuracy. The alternate
settings were explored on this same clip, not validated on a held-out dataset.

`out/IMG_2422/IMG_2422_comparison.mp4` shows standard settings on the left and
subpixel refinement disabled on the right. `summary.json` contains all seven
runs; `id0_frame_0062.png` shows the four-column detection. Both exported MP4s
were decoded and checked to contain all 266 frames, and all seven CSV/JSONL
pairs were checked for matching frame counts, marker counts, and ID 0 flags.

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
