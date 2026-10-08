# Release validation

Checks performed on 2026-10-08, using Ubuntu 26.04, Python 3.11.16,
PyTorch 2.1.0+cpu, Python OpenCV 4.11.0, and system OpenCV 4.10.0 for C++.

## Student package

```bash
python scripts/check_setup.py
python scripts/check_starter.py
python -m pytest tests/test_helpers.py tests/test_court_runner.py tests/test_weight_download.py tests/test_team_training.py
```

The setup check passed. All 43 exercise IDs have tests. The starter had 37 passing
infrastructure tests and 108 intentional `NotImplementedError` failures, summarized
as 41 TODO and 2 BLOCKED exercise IDs. There were no unexpected failures, collection
errors, or skipped tests. Exercise filtering with `--ex 1.4` also worked.

All 43 exercise TODOs remain in the student code. No solutions or fixture changes
were added to the student package.

## Completed exercise check

A separate instructor copy was reconstructed from the original
`BadmintonAI-End2End` implementations. The original implementations required
corrections for single-frame inference batches, tail-frame bookkeeping, padded
window lengths, homography axes, and missing-shuttle identity handling. The new
player-box helper was implemented according to its exercise contract.

The completed copy is now preserved outside the student tree, in a private instructor
directory. Ex 3.9 also rejects invisible or non-finite shuttle observations, absent
or invalid boxes, and incomplete first-frame player detections.

After those corrections, all 145 tests passed, including all 43 exercise IDs.
This verifies that the exercise contracts and fixtures can be satisfied together.
It does not establish that the unmodified original repository is a passing answer key.

## Native court detector

```bash
bash scripts/build_court_detector.sh
python scripts/smoke_test_court.py
```

The build produced a native x86-64 Linux executable. Both the image and generated
MP4 cases passed. Maximum outer-corner differences against the supplied reference
were 9.0 and 9.6 pixels respectively. The camera-to-court homography projected the
four corners to the upstream model coordinates within 0.001 court units.
Missing-input and out-of-range-frame cases returned errors. The Python runner was
also exercised with the actual executable and generated MP4.

The input is one already-annotated image. These checks are regression evidence,
not a court-detection accuracy benchmark. Inspect the generated overlays and test
representative unannotated match frames before class-wide video integration.

## Model assets

`python scripts/fetch_weights.py --check` passed for all five default pipeline
checkpoints: HitNet, TemPose, TrackNet, the rally classifier, and YOLO. The missing
LFS payloads were downloaded and checked against pinned SHA-256 values. Optional
training checkpoints may still be LFS pointers; use `--all` to fetch those if needed.

HitNet, TrackNet, TemPose, and the rally classifier loaded with strict state-dict
matching into the repository model classes on the CPU. The YOLO file is a valid
Ultralytics checkpoint archive; Ultralytics was not installed, so it was not loaded.

## Match data

The eight label files in `data/shuttleset/` were matched to ShuttleSet by comparing
frame numbers with every match's set files. `full_35` to `full_39` are matches 35 to 39,
`full_1` is match 29, `full_2` is match 21, and `full_3` is match 14. Each file is the
concatenation of all sets of its match, with frame numbers from the full raw video.
The videos in `data/videos/` are copies of those raw videos. OpenCV 4.11 decoded a
middle frame of all eight. Match 39 was AV1, which OpenCV could not decode, so it was
re-encoded to H.264 with every frame kept. `scripts/evaluate.py` was run on synthetic
predictions built from the labels, including swapped team ids and unknown players.

## Limits

The complete GPU pipeline was not run on any match video in this CPU exercise
environment. The instructor must run the chosen class video through
a completed solution and record expected outputs before assigning that extension.
Build the detector on each target Ubuntu release instead of assuming that this
machine's executable is binary-compatible with older releases.

## Review fixes and current pipeline preflight

A fresh Python 3.11.16 environment installed CPU PyTorch 2.1.0 and the exercise
requirements. The setup and starter checks passed. Added evaluation regressions
cover multi-hit intervals, duplicate predictions, empty inputs, input ordering,
invalid frames, and player mapping. A synthetic plotting run also exercised report
and image generation, including empty inputs. These are synthetic checks, not match
accuracy measurements.

The release builder excludes bulk assets and refuses completed solutions. The
student ZIP is extracted and checked independently before delivery. See RELEASE.md.

The current machine reports an NVIDIA RTX 5090, but the exercise environment has
CPU-only PyTorch and lacks MMPose, MMCV, MMDetection, Ultralytics, and other pipeline
dependencies. Its pipeline preflight fails. No full-video result has been produced;
the optional extension remains unvalidated until the release gate in RELEASE.md is
completed on a configured pipeline environment.
