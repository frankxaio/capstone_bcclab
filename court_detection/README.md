# Build and run court detection on Ubuntu

This directory contains C++ source that builds a native Linux executable with OpenCV 4.
No Windows binary is included. On Windows, supply court coordinates with `--court-file`.
The detector runs on the CPU and needs no model weights or display server.

## Build

From the repository root:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake libopencv-dev
bash scripts/build_court_detector.sh
```

The output is `court_detection/build/court-detection`. Build on the target machine;
a binary built on a newer Ubuntu release might not run on an older release.

## Run

```bash
court_detection/build/court-detection match.mp4 /tmp/court.txt /tmp/court.png 100
```

The last argument is a zero-based frame index. Omit it to use the middle frame.
You can also supply a still image with no frame index:

```bash
court_detection/build/court-detection court_detection/court_image.png /tmp/court.txt /tmp/court.png
```

Inspect the overlay before using the coordinates. The included image already has
annotations and is only a smoke-test input, not an accuracy benchmark.
The detector expects a broadcast view with visible court lines. Camera changes,
occlusion, and non-court frames can produce bad fits. Full-pipeline calls limit
attempts and time out; direct calls to the executable do not impose a time limit.

## Verify the build

In the CPU exercise environment:

```bash
python scripts/smoke_test_court.py
```

This runs the actual executable on the bundled image and an MP4 generated from
that image. It checks corner agreement with the supplied reference within 25 pixels,
verifies that the homography projects corners to the upstream court model, and
checks missing-input and out-of-range-frame errors. It saves overlays under
`court_detection/build/smoke-test/`. This is a regression test on one annotated
frame, not a measure of accuracy on new matches.

## Use in the pipeline

```bash
python scripts/run_pipeline.py match.mp4 --court-attempts 5 --court-timeout 120
```

The runner chooses readable rally clips and valid frame indices, checks the exit
status and output format, and applies your Ex 1.3 corner check. It never reuses
output from a failed attempt. It uses `court_detection/build/court-detection` by default.
Override it with `--court-executable /path/to/court-detection`.

If you already have verified coordinates for the same video resolution and camera:

```bash
python scripts/run_pipeline.py match.mp4 --court-file /path/to/verified-court.txt
```

Do not use the bundled `court.txt` for an unrelated match. Supplying a court file
skips automatic detection, but still requires a valid file and completed Ex 1.3.
It does not generate a new detector overlay.

## Output contract and source

The output contains six `x;y` lines, ordered top-left, bottom-left, bottom-right,
top-right, left net pole, right net pole. These are followed by `Homography Matrix:`
and a bracketed 3-by-3 camera-to-court matrix. Court coordinates use the upstream
6.1 by 13.41 model. Downstream normalisation uses the transformed court corners.

Source: [MonoTrack court detector](https://github.com/jhwang7628/monotrack/tree/39e9227bd44fa2a579731b1cc97c7f529f391cd6/court-detection),
revision `39e9227bd44fa2a579731b1cc97c7f529f391cd6`. The BSD notice is in [LICENSE](LICENSE).
Local changes replace the build with system OpenCV 4, update OpenCV constants,
use BGR luminance conversion, add a headless image/video CLI with frame validation,
append the inverse fitted homography for this pipeline, reject off-frame court fits,
fix an incorrectly sized net line-pair table, and skip empty random line partitions
without discarding an earlier valid fit. Results may differ
from the Windows binary that the original project shipped.
