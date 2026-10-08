# Set up the full video pipeline

Complete the CPU exercises first. This integration environment is separate from the
lightweight exercise environment. The complete pipeline still needs an NVIDIA GPU
for its explicit CUDA MMPose call. Native court detection itself runs on the CPU.

## Install the Python environment

Use Python 3.11 and a separate virtual environment. Run from the repository root:

```bash
python3.11 -m venv .venv-pipeline
source .venv-pipeline/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
mim install mmcv==2.1.0
python -m pip install pytest==9.1.1
```

On Windows, activate with `.venv-pipeline\Scripts\Activate.ps1` in PowerShell.
The CUDA wheel includes its runtime; the machine still needs a compatible NVIDIA
driver. MMCV contains compiled operators. If its installation starts a source build,
use an instructor-provided environment or resolve the compiler/CUDA requirements
before continuing. The full dependency installation is not needed for CPU exercises.

The pinned `requirements.txt` is the original full-pipeline dependency list, now
UTF-8. It is not a cross-platform lockfile. Do not treat a successful CPU exercise
run as validation of this heavier environment.

## Obtain weights

```bash
python scripts/fetch_weights.py
python scripts/fetch_weights.py --check
```

The downloader works with a ZIP/folder delivery without Git. It retrieves only the
five checkpoints used by the default pipeline, verifies size and SHA-256 against
`scripts/weights.json`, and replaces a file only after successful verification.
Rerunning skips valid files. Network access is required for missing weights.
Use `--all` only if you need the optional training checkpoints, about 792 MB total.
The manifest pins the original repository revision; no latest-model lookup is used.

This student repository does not track checkpoints and does not require Git LFS.
Use the downloader above after cloning. If you obtain weights from an upstream
repository, a file beginning with `version https://git-lfs.github.com/spec/v1` is
only a pointer, not a usable model.
MMPose and the team classifier may additionally download
upstream models on first use; prepare the class machine with network access.

## Build court detection on Ubuntu

Follow [court detector instructions](../court_detection/README.md). No Windows binary
is included. On Windows, or instead of building the detector, provide verified
coordinates for the same video and camera with `--court-file`.

```bash
python scripts/check_setup.py --pipeline
# Or, when using supplied court coordinates:
python scripts/check_setup.py --pipeline --court-file /path/to/verified-court.txt
```

This checks imports, checkpoint hashes, a CUDA computation, and court assets.
Run it on the intended class GPU; installing a wheel and detecting the driver alone
do not prove that the wheel can execute on that GPU.
It does not validate prediction accuracy or replace an actual video run.

## Run a match

Download the class videos from [Google Drive](https://drive.google.com/drive/folders/1YFi0mZLx8A9gnoQQ49Wjc4EhjQBkuWDb?usp=drive_link)
and put them in `data/videos/` as described in [match data](../data/README.md).
You can also use an instructor-approved video. Start
with singles badminton at 1280x720: some existing feature code and the player ROI
use fixed image coordinates. Other resolutions and changing camera views require
additional work beyond the exercise TODOs.

```bash
python scripts/run_pipeline.py /path/to/match.mp4
# With verified court coordinates:
python scripts/run_pipeline.py /path/to/match.mp4 --court-file /path/to/court.txt
```

The runner validates weights and court assets before loading the heavy pipeline.
Court detection tries up to five frames, each with a 120-second timeout. Override
with `--court-attempts` and `--court-timeout`. No `clip_7` dependency remains.
Team classification uses the court detected for that video as its player ROI.
It samples player crops for at most five rounds, warns below its 3,500-crop target,
and stops with an error below 100 crops instead of retrying indefinitely.

Intermediate clips and features go to `videos/<video-stem>/`. Final outputs are
`results/<video-stem>/hit_events.csv`, `summary_counts.csv`, and an annotated video.
Use distinct video stems for distinct inputs; existing inference caches use the stem.
The original rally clipper skips its first detected rally, so use a match with several
rallies rather than a single-rally snippet for this extension.

## Evaluate against ShuttleSet labels

`data/shuttleset/` holds ShuttleSet stroke labels for eight matches, and
[match data](../data/README.md) explains which video goes with each file. Run the
pipeline on a video from `data/videos/`, then evaluate its results:

```bash
python scripts/run_pipeline.py data/videos/full_36.mp4
python scripts/evaluate.py full_36
```

The script compares `results/full_36/hit_events.csv` with the labels. It prints hit
detection, player-and-stroke, and stroke distribution metrics, and saves confusion
matrices and reports in `results/full_36/`. The pipeline's team ids 0 and 1 come from
clustering and can swap between runs, so the script picks the mapping from team id to
ShuttleSet player that agrees with more labelled hits, and prints it.

### How hit metrics are computed

A prediction interval includes both its start and end frame. It is eligible only
when it contains exactly one ground-truth hit. Each prediction and ground-truth row
can match at most once. When predictions compete for one hit, the shortest interval
wins, followed by the earliest start and input order. An interval covering multiple
true hits is a false positive; those true hits remain misses unless other eligible
predictions match them.

The report gives TP, FP, FN, precision, recall, and F1, and saves them to
`hit_detection_metrics.csv`. It saves a count chart as `hit_event_counts.png`.
There is no event-level true-negative count or detection accuracy because arbitrary
background gaps are not comparable to hit events. Older `hit_events_confusion_matrix.png`
files are obsolete and are not regenerated by this evaluator.

Player mapping and stroke classification use the same matches. Classification
accuracy and macro F1 describe matched events only; always report hit recall and the
unmatched counts alongside them. Empty inputs produce zero scores. Stroke-distribution
KL excludes `unknown` from both the distributions and the normalized uniform baseline.

## Instructor release check

Before assigning the optional integration milestone:

1. In the student copy, run `python scripts/check_setup.py` and
   `python scripts/check_starter.py`. The latter permits only intentional TODO
   failures and rejects unexpected failures, collection errors, or skipped tests.
   Then run all tests on the instructor's completed solution.
2. Run the same video that students will receive through the whole pipeline.
3. Inspect the court overlay and manually check representative output events.
4. Record OS, GPU/driver, package versions, runtime, and expected output files.
5. Confirm students can access the video and model downloads.

Keep the instructor solution separate from the student package. The supplied
fixtures test deterministic intermediate computations; they are not a full-match
reference result.
