# Capstone exercise assignment

Implement the 43 labelled exercise blocks so the system can turn badminton video
features into hit times, stroke labels, and player identities.

## Prerequisites

You need Python functions and classes, NumPy indexing and broadcasting, basic
linear algebra, and PyTorch tensors, modules, losses, and back-propagation.
Review convolution, recurrent networks, and attention before the model milestone.
The required track uses a CPU. Full video integration needs the separate pipeline
setup and an NVIDIA GPU. The instructor sets calendar deadlines and team sizes.

## Required milestones

| Milestone | Work | Evidence to submit |
|---|---|---|
| 1. Environment and warm-up | Set up the CPU environment; Ex 1.3, 3.10a, 3.8a, 2.5a, 3.3a, 3.1a | Setup-check output and selected pytest results |
| 2. Data processing | Remaining data exercises in the README suggested order | Test results and an explanation of coordinate systems, frame offsets, and padding |
| 3. Models and training | Attention, TrackNet, HitNet, TemPose, losses, and training steps | Test results; trace the input/output shapes through one model |
| 4. Integration logic | Finish all 43 exercise IDs, including identity and event merging | Full pytest output and a failure analysis on a synthetic edge case |

Use the README suggested order within each milestone. A BLOCKED result identifies
an exercise dependency; finish that prerequisite before debugging the dependent test.
The starter intentionally fails exercise tests. Do not remove tests, modify fixtures,
rename pretrained layers, or hard-code fixture outputs to make them pass.

## Submission

Submit your source changes, `python -m pytest` output, and a short report containing:

- Python/package versions and exact reproduction commands.
- Milestone results and any remaining TODO, BLOCKED, FAIL, or SKIP statuses.
- One worked example tracing frame numbers and tensor shapes across stages.
- Two edge cases you tested and why they matter.
- Individual contributions for team submissions and sources you used.

Do not include environments, downloaded weights, or broadcast videos in your submission.
Keep the original test suite and fixtures intact.

## Grading rubric

| Criterion | Points |
|---|---:|
| Correctness across exercise IDs, assessed with tests and code review | 50 |
| Explanation of shapes, coordinate systems, temporal alignment, and model behaviour | 20 |
| Edge-case analysis and additional meaningful checks | 15 |
| Reproducible setup and readable implementation | 10 |
| Report, attribution, and contribution record | 5 |

Correctness is assessed by exercise ID, not raw parametrized test count. A SKIP is
not a pass. Public tests provide feedback; passing them alone does not demonstrate
that the full video pipeline works.

## Optional full-video extension

After the required track passes, use [pipeline setup](PIPELINE.md) and demonstrate
one labelled match from [match data](../data/README.md), or another instructor-approved
singles broadcast with stable camera geometry. Submit `hit_events.csv`,
`summary_counts.csv`, an annotated excerpt, runtime and hardware, and the output of
`python scripts/evaluate.py` for a labelled match. Also manually check a small set of
hit times, stroke labels, and identities.
Report observed errors rather than claiming accuracy from the unit tests.
The instructor may assign extension credit separately; it is not part of the
100-point required rubric above.

## Reading

The repository implementations include local modifications. Follow each TODO's
contract when it differs from a paper. No files in the parent directory are required.

- [Exploration of Player Behaviours from Broadcast Badminton Videos](https://diglib.eg.org/items/cb042e27-f11c-4bca-881b-f6ea9709bc27), rally perception context.
- [TrackNetV3 author paper](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf), shuttle tracking context. This repo's CBAM variant is described in `TrackNetV3/README.md`.
- [MonoTrack](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Liu_MonoTrack_Shuttle_Trajectory_Reconstruction_From_Monocular_Badminton_Video_CVPRW_2022_paper.pdf), court and trajectory reconstruction context.
- [TemPose](https://openaccess.thecvf.com/content/CVPR2023W/CVSports/html/Ibh_TemPose_A_New_Skeleton-Based_Transformer_Model_Designed_for_Fine-Grained_Motion_CVPRW_2023_paper.html), temporal and player-interaction modelling.
