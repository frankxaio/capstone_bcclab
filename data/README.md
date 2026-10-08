# Match data

The required CPU exercises do not use this folder. It holds the match videos and labels
for the optional full-video run and its evaluation.

```
data/
├── shuttleset/
│   ├── full_1.csv ... full_39.csv   Stroke labels, one file per match
│   └── matches.csv                  Which match each file labels
└── videos/
    └── full_1.mp4 ... full_39.mp4   Broadcast videos, shared separately
```

## Matches

| Name | Match | Player A | Player B | FPS |
|---|---|---|---|---|
| `full_1` | TOYOTA Thailand Open 2021, quarter-final | An Se Young | Pornpawee Chochuwong | 30 |
| `full_2` | YONEX Thailand Open 2021, quarter-final | An Se Young | Ratchanok Intanon | 30 |
| `full_3` | Indonesia Masters 2020, quarter-final | Anders Antonsen | Jonatan Christie | 25 |
| `full_35` | TOYOTA Thailand Open 2021, semi-final | Carolina Marin | An Se Young | 30 |
| `full_36` | TOYOTA Thailand Open 2021, semi-final | Hans-Kristian Solberg Vittinghus | Anders Antonsen | 30 |
| `full_37` | TOYOTA Thailand Open 2021, final | Viktor Axelsen | Hans-Kristian Solberg Vittinghus | 30 |
| `full_38` | BWF World Tour Finals 2020, quarter-final | An Se Young | Carolina Marin | 30 |
| `full_39` | BWF World Tour Finals 2020, quarter-final | Anthony Sinisuka Ginting | Lee Zii Jia | 30 |

Player A won the match. `matches.csv` also lists the ShuttleSet match ID, the frame count
of each video, and its YouTube source. `full_1` is the original project's sample match.

## Labels

Each label file covers every set of one match. A row is one shot: its rally, the frame
number of the hit, the player, the shot type, and the court positions. Frame numbers
count from the first frame of the full broadcast video, so a label file only lines up
with its own video. The shot types use ShuttleSet's English names, plus `unknown`.

## Videos

The videos are about 4.8 GB in total, so they are not part of the code package. Download them from the
[class Google Drive folder](https://drive.google.com/drive/folders/1YFi0mZLx8A9gnoQQ49Wjc4EhjQBkuWDb?usp=drive_link). If Drive requests access, sign in with your
class account and request access from the instructor. Put the downloaded `.mp4` files
in `data/videos/` with the names shown above, then check that each video's frame count
matches `matches.csv`:

```bash
python -c "import cv2, sys; print(int(cv2.VideoCapture(sys.argv[1]).get(cv2.CAP_PROP_FRAME_COUNT)))" data/videos/full_1.mp4
```

A different download, or a trimmed copy, shifts the frame numbers and breaks the
evaluation. `full_39.mp4` was converted from AV1 to H.264 because OpenCV cannot decode
AV1 on Ubuntu. The conversion kept every frame. The videos are BWF broadcast recordings.
Share them only within the class.

## Run and evaluate

```bash
python scripts/run_pipeline.py data/videos/full_1.mp4
python scripts/evaluate.py full_1
```

The second command compares `results/full_1/hit_events.csv` with `data/shuttleset/full_1.csv`.
See [pipeline setup](../docs/PIPELINE.md#evaluate-against-shuttleset-labels) for details.

## Source

The labels come from ShuttleSet, a human-annotated stroke-level dataset of singles
badminton. This course package translated the shot types and rally-end reasons from
Chinese to English. Cite the dataset if you use it in a report:

```
@article{ShuttleSet,
  author  = {Wei{-}Yao Wang and Yung{-}Chang Huang and Tsi{-}Ui Ik and Wen{-}Chih Peng},
  title   = {ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for Badminton Tactical Analysis},
  journal = {CoRR},
  volume  = {abs/2306.04948},
  year    = {2023}
}
```
