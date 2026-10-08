# Distribute the student exercises

The CPU practice track is ready for classroom use. The optional full-video track
remains experimental until an instructor validates the selected match on a working
GPU pipeline environment. Students should not need to debug that environment as part
of the required exercises.

## Build the student archive

From the student source directory:

```bash
python scripts/check_setup.py
python scripts/check_starter.py
python scripts/package_student.py
```

The default output is `../releases/BadmintonAI-student.zip`, with a SHA-256 sidecar.
Use `--output /path/to/student.zip` to choose another location. The builder preserves
all 43 exercise TODOs, tests, fixtures, documentation, licenses, label CSVs, and court
source. It excludes videos, model checkpoints, caches, native build outputs, and
unlisted top-level folders. It refuses to package a completed solution as a starter.
The ZIP has a stable file order and timestamps, so rebuilding unchanged files gives
the same checksum.

Extract the archive into a new directory and run the setup and starter checks there
before sharing it. Share videos separately for the optional extension; model weights
are obtained with `scripts/fetch_weights.py`.

Keep the completed instructor solution outside the student source and archive.
Retain a private backup of that solution and run its full test suite whenever the
assignment or tests change. The original upstream implementation needs corrections
and should not be used as an answer key without validation.

## Full-video release gate

On the intended class machine, using a completed solution:

1. Follow `PIPELINE.md` to create a separate pipeline environment and obtain assets.
2. Run `python scripts/check_setup.py --pipeline`. The check executes a CUDA
   computation as well as checking imports and assets. A driver reporting a GPU is
   not sufficient. Do not continue if this check fails.
3. Run `python scripts/run_pipeline.py data/videos/full_1.mp4` and then
   `python scripts/evaluate.py full_1`.
4. Inspect the court overlay and an annotated excerpt. Compare representative
   hit times, stroke labels, and player identities with the supplied labels.
5. Save package versions, GPU/driver, runtime, expected output files, and observed
   errors alongside the private instructor materials.

No full-match accuracy or runtime is certified by the CPU test results.
