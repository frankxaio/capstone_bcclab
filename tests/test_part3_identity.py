"""Process III - player identity and hit events (Ex 3.8 - 3.10)."""
import shutil
import types

import numpy as np
import pytest

from _helpers import ROOT, import_repo, require, write_video


# --------------------------------------------------------------------------- Ex 3.8
def team_module(monkeypatch):
    mod = import_repo('team_classifier.sport_player_team_classifier')
    for name, value in zip(['ROI_X1', 'ROI_Y1', 'ROI_X2', 'ROI_Y2'], [100, 100, 500, 400]):
        monkeypatch.setattr(mod, name, value)
    return mod


@pytest.mark.ex('3.8a')
@pytest.mark.parametrize('box, expected', [
    ((200, 150, 300, 300), True),     # inside
    ((450, 350, 600, 500), True),     # partly inside
    ((0, 0, 1000, 1000), True),       # covers the whole ROI
    ((0, 150, 90, 300), False),       # left of the ROI
    ((510, 150, 600, 300), False),    # right
    ((200, 0, 300, 90), False),       # above
    ((200, 410, 300, 500), False),    # below
])
def test_box_overlaps_roi(monkeypatch, box, expected):
    mod = team_module(monkeypatch)
    assert bool(mod.box_overlaps_roi(*box)) is expected


DETECTIONS = [  # (xyxy, confidence, kept?)
    ((200.0, 150.0, 300.0, 300.0), 0.90, True),
    ((210.0, 160.0, 310.0, 310.0), 0.50, False),      # not confident
    ((600.0, 150.0, 700.0, 300.0), 0.95, False),      # outside the court ROI
    ((220.0, 160.0, 240.0, 180.0), 0.80, False),      # too small (a racket)
    ((350.7, 200.2, 450.9, 380.5), 0.60, True),       # confidence == CONF_THRESH is kept
    ((100.1, 100.1, 160.9, 150.9), 0.99, False),      # int corners -> area 60 * 50 = 3000, not > 3000
]


def detections(rows):
    return types.SimpleNamespace(xyxy=np.array([r[0] for r in rows], dtype=float).reshape(-1, 4),
                                 confidence=np.array([r[1] for r in rows], dtype=float))


@pytest.mark.ex('3.8b')
def test_select_player_boxes(monkeypatch):
    mod = team_module(monkeypatch)
    kept = mod.select_player_boxes(detections(DETECTIONS))
    assert isinstance(kept, list), 'return a list of boxes'
    np.testing.assert_allclose(np.array(kept, dtype=float).reshape(-1, 4), [r[0] for r in DETECTIONS if r[2]],
                               err_msg='keep confident (>= CONF_THRESH), on-court, big enough boxes - unchanged')
    assert mod.select_player_boxes(detections([])) == []


@pytest.mark.ex('3.8b')
def test_predict_teams_writes_kept_boxes(tmp_path, monkeypatch):
    mod = team_module(monkeypatch)
    require('pandas')
    import pandas as pd
    clip_dir = tmp_path / 'clip_1'
    clip_dir.mkdir()
    write_video(clip_dir / 'clip_1.mp4', [np.zeros((64, 64, 3), np.uint8)] * 3)
    per_frame = [DETECTIONS[:2], DETECTIONS[2:4], [DETECTIONS[0], DETECTIONS[4]]]

    class FakeYOLO:
        def __init__(self, weights):
            self.frame = -1

        def to(self, device):
            return self

        def __call__(self, frame, imgsz=640, verbose=False):
            self.frame += 1
            return [self.frame]

    class FakeClassifier:
        def predict(self, crops):
            return [7 + i for i in range(len(crops))]

    monkeypatch.setattr(mod, 'YOLO', FakeYOLO)
    monkeypatch.setattr(mod, 'sv', types.SimpleNamespace(
        Detections=types.SimpleNamespace(from_ultralytics=lambda frame: detections(per_frame[frame])),
        crop_image=lambda image, xyxy: image))
    mod.predict_teams(str(clip_dir), 'clip_1', FakeClassifier(), 100, draw=False)
    rows = pd.read_csv(clip_dir / 'clip_1_teams.csv').values.tolist()
    assert rows == [[100, 0, 200, 150, 300, 300, 7],
                    [102, 0, 200, 150, 300, 300, 7],
                    [102, 1, 350, 200, 450, 380, 8]]


# --------------------------------------------------------------------------- Ex 3.9
class _Stop(Exception):
    pass


def write_csv(path, header, rows):
    path.write_text(','.join(header) + '\n' + '\n'.join(','.join(map(str, r)) for r in rows) + '\n')


def run_main(tmp_path, monkeypatch, hits):
    """Run main() with every pipeline stage faked; return the hit table built by Ex 3.9."""
    require('pandas', 'yaml', 'torch')
    main = import_repo('main')
    import torch
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'TemPose' / 'config').mkdir(parents=True)
    shutil.copy(ROOT / 'TemPose' / 'config' / 'config.yml', tmp_path / 'TemPose' / 'config' / 'config.yml')
    videos = tmp_path / 'videos' / 'full_36'
    team_header = ['frame', 'player_id', 'x1', 'y1', 'x2', 'y2', 'team_id']
    # clip_1: team 1 plays at the top, team 0 at the bottom
    (videos / 'clip_1').mkdir(parents=True)
    write_csv(videos / 'clip_1' / 'clip_1_teams.csv', team_header, [
        [100, 0, 500, 150, 560, 300, 1], [100, 1, 300, 400, 380, 600, 0],
        [105, 0, 500, 150, 560, 300, 1], [105, 1, 300, 400, 380, 600, 0],
        [110, 0, 510, 160, 570, 310, 1], [110, 1, 310, 410, 390, 610, 0]])
    write_csv(videos / 'clip_1' / 'clip_1_ball.csv', ['Frame', 'Visibility', 'X', 'Y'],
              [[f, 1, 530, 200] if f == 105 else [f, 1, 345, 520] if f == 110 else [f, 0, 0, 0]
               for f in range(100, 125) if f != 115])       # no shuttle row for frame 115
    # clip_3: after the break the teams switched sides
    (videos / 'clip_3').mkdir(parents=True)
    write_csv(videos / 'clip_3' / 'clip_3_teams.csv', team_header, [
        [500, 0, 300, 420, 380, 600, 1], [500, 1, 500, 140, 560, 300, 0]])
    write_csv(videos / 'clip_3' / 'clip_3_ball.csv', ['Frame', 'Visibility', 'X', 'Y'],
              [[f, 0, 0, 0] for f in range(500, 530)])
    for clip in ('clip_1', 'clip_3'):
        write_csv(videos / clip / f'{clip}_hits.csv', ['frame', 'hit'], [[0, 0]])

    class FakeTemPose:
        def __init__(self, **kwargs):
            pass

        def to(self, device):
            return self

        def load_state_dict(self, state):
            pass

        def eval(self):
            return self

    captured = {}

    def capture(df_events):
        captured['events'] = df_events
        raise _Stop

    def centre_distance(ball, bbox):   # any distance to the box centre works for these tests
        return max(abs(ball[0] - (bbox[0] + bbox[2]) / 2), abs(ball[1] - (bbox[1] + bbox[3]) / 2))

    fakes = {
        'get_video_fps': lambda path: 30,
        'timepoints_clipping': lambda path, fps: {'clip_1': 100, 'clip_3': 500},
        'detect_court': lambda *a, **k: None,
        'check_court_coordinates': lambda path: True,
        'load_tracknet_model': lambda: None,
        'MMPoseInferencer': lambda *a, **k: None,
        'predict_traj': lambda *a, **k: 'ball.csv',
        'process_pose': lambda *a, **k: None,
        'hitnet_detect': lambda *a, **k: None,
        'train_yolo': lambda *a, **k: None,
        'predict_teams': lambda *a, **k: None,
        'process_clip': lambda *a, **k: None,
        'TemPoseII_TF': FakeTemPose,
        'per_hit_predict': lambda model, clip_dir, hit_csv, cfg, device: hits[clip_dir.name],
        'calculate_ball_bbox_dist': centre_distance,
        'merge_consecutuve_frame': capture,
    }
    for name, fake in fakes.items():
        monkeypatch.setattr(main, name, fake)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: {'model_state_dict': {}})
    with pytest.raises(_Stop):
        main.main()
    df = captured['events']
    return [(int(f), s, p if p == 'X' else int(p)) for f, s, p in zip(df['frame'], df['stroke'], df['player'])]


@pytest.mark.ex('3.9')
def test_who_hit_the_shot(tmp_path, monkeypatch):
    hits = {'clip_1': ([101, 105, 110, 120], ['Top_smash', 'unknown', 'unknown', 'Bottom_drop']),
            'clip_3': ([510, 520], ['Top_clear', 'Bottom_lob'])}
    assert run_main(tmp_path, monkeypatch, hits) == [
        (101, 'Top_smash', 1),       # Top_ -> the team at the top in the clip's first frame
        (105, 'unknown', 1),       # unknown -> the team whose box is closest to the shuttle
        (110, 'unknown', 0),
        (120, 'Bottom_drop', 0),
        (510, 'Top_clear', 0),       # clip_3: the teams switched sides
        (520, 'Bottom_lob', 1),
    ]


@pytest.mark.ex('3.9')
def test_hit_without_shuttle_position(tmp_path, monkeypatch):
    hits = {'clip_1': ([101, 115, 120], ['Top_smash', 'unknown', 'Bottom_drop']), 'clip_3': ([], [])}
    assert run_main(tmp_path, monkeypatch, hits) == [
        (101, 'Top_smash', 1),
        (115, 'unknown', 'X'),     # no shuttle row -> player unknown, but the hit is kept
        (120, 'Bottom_drop', 0),
    ]


# --------------------------------------------------------------------------- Ex 3.10
@pytest.mark.ex('3.10a')
@pytest.mark.parametrize('ball, bbox, expected', [
    ((3, 4), (0, 0, 2, 4), 8.0),           # centre (1, 2): 2^2 + 2^2
    ((10, 10), (10, 10, 10, 10), 0.0),
    ((0, 0), (2, 4, 4, 8), 45.0),          # centre (3, 6): 3^2 + 6^2
])
def test_ball_bbox_distance(ball, bbox, expected):
    mod = import_repo('utils')
    assert mod.calculate_ball_bbox_dist(ball, bbox) == pytest.approx(expected)


U = 'unknown'


def merge(rows):
    require('pandas')
    import pandas as pd
    mod = import_repo('utils')
    events = pd.DataFrame(rows, columns=['frame', 'stroke', 'player'])
    merged, counts = mod.merge_consecutuve_frame(events)
    return ([(int(a), int(b), s, p if p == 'X' else int(p)) for a, b, s, p in merged.itertuples(index=False)],
            dict(zip(counts['stroke'], counts['count'])))


@pytest.mark.ex('3.10b')
def test_merge_hit_events():
    rows = ([(f, 'A', 1) for f in (10, 11, 12)] + [(f, 'A', 1) for f in (15, 16)]      # same stroke & player, gap 3
            + [(40, U, 'X'), (41, U, 'X'), (45, 'B', 0), (46, 'B', 0)]                 # unknown + known, gap 4
            + [(80, 'C', 1), (81, 'C', 1), (85, 'D', 0)]                               # different strokes
            + [(120, 'C', 1), (121, 'C', 1), (150, 'C', 1), (151, 'C', 1)]             # gap 29: too far apart
            + [(200, 'E', 1), (205, U, 0)]                                             # known + unknown
            + [(300, 'F', 0), (305, 'F', 1)])                                          # same stroke, other player
    events, counts = merge(rows)
    assert events == [(10, 16, 'A', 1), (40, 46, 'B', 0), (80, 81, 'C', 1), (85, 85, 'D', 0),
                      (120, 121, 'C', 1), (150, 151, 'C', 1), (200, 205, 'E', 1), (300, 300, 'F', 0),
                      (305, 305, 'F', 1)]
    assert counts == {'A': 1, 'B': 1, 'C': 3, 'D': 1, 'E': 1, 'F': 2}


@pytest.mark.ex('3.9')
@pytest.mark.parametrize('case', ['invisible', 'nonfinite', 'no_boxes', 'invalid_boxes', 'empty_teams', 'one_player'])
def test_identity_without_usable_observations(tmp_path, monkeypatch, case):
    require('pandas')
    import pandas as pd
    original = pd.read_csv

    def read_csv(path, *args, **kwargs):
        df = original(path, *args, **kwargs)
        if str(path).endswith('clip_1_ball.csv'):
            if case == 'invisible':
                df.loc[df['Frame'] == 105, ['Visibility', 'X', 'Y']] = [0, 0, 0]
            elif case == 'nonfinite':
                df.loc[df['Frame'] == 105, 'X'] = float('nan')
        elif str(path).endswith('clip_1_teams.csv'):
            if case == 'no_boxes':
                df = df[df['frame'] != 105]
            elif case == 'invalid_boxes':
                df.loc[df['frame'] == 105, ['x1', 'y1', 'x2', 'y2']] = 0
            elif case == 'empty_teams':
                df = df.iloc[:0]
            elif case == 'one_player':
                df = df[df['team_id'] == 1]
        return df

    monkeypatch.setattr(pd, 'read_csv', read_csv)
    stroke = 'Top_smash' if case in ('empty_teams', 'one_player') else 'unknown'
    hits = {'clip_1': ([105], [stroke]), 'clip_3': ([], [])}
    assert run_main(tmp_path, monkeypatch, hits) == [(105, stroke, 'X')]
