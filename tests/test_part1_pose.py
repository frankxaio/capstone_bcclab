"""Process I - keep only the two players among the detected people (Ex 1.7)."""
import numpy as np
import pytest

from _helpers import import_repo, require

COURT = np.array([[400, 300], [300, 650], [980, 650], [870, 300]], dtype=np.int32)   # TL, BL, BR, TR


def pose_module():
    return import_repo('MMPose.detect_pose', 'cv2')


# --------------------------------------------------------------------------- Ex 1.7a
@pytest.mark.ex('1.7a')
@pytest.mark.parametrize('heels, threshold, expected', [
    (((600.0, 500.0), (620.0, 505.0)), 15, True),     # both heels inside
    (((600.0, 500.0), (600.0, 900.0)), 15, True),     # one heel inside is enough
    (((600.0, 660.0), (620.0, 662.0)), 15, True),     # 10-12 px outside: within the threshold
    (((600.0, 680.0), (620.0, 690.0)), 15, False),    # 30-40 px outside
    (((600.0, 680.0), (620.0, 690.0)), 50, True),     # ... unless the threshold is larger
    (((100.0, 400.0), (110.0, 400.0)), 15, False),    # umpire next to the court
])
def test_is_inside_court(heels, threshold, expected):
    mod = pose_module()
    assert bool(mod.is_inside_court(heels, COURT, threshold=threshold)) is expected


# --------------------------------------------------------------------------- Ex 1.7b / 1.7c
def person(x, heel_y):
    """A detected person: 17 COCO keypoints (heels = keypoints 15, 16) and a bounding box."""
    kps = [[x + 2.0 * j, heel_y - 150.0 + 5.0 * j] for j in range(17)]
    kps[15], kps[16] = [x - 10.0, heel_y], [x + 10.0, heel_y]
    return {'keypoints': kps, 'bbox': ([x - 30.0, heel_y - 160.0, x + 30.0, heel_y + 5.0],)}


class FakeCapture:
    def __init__(self, n):
        self.n, self.i = n, 0

    def read(self):
        if self.i < self.n:
            self.i += 1
            return True, np.zeros((72, 128, 3), np.uint8)
        return False, None

    def get(self, prop):
        return 30.0

    def release(self):
        pass


def run_pose(tmp_path, monkeypatch, frames):
    mod = pose_module()
    require('pandas')
    import pandas as pd

    def inferencer(images, show=False):
        assert len(images) == len(frames)
        for people in frames:
            yield {'predictions': [people]}

    # "inside the court" = heels between y=250 and y=700 (Ex 1.7a is tested separately)
    monkeypatch.setattr(mod, 'is_inside_court', lambda feet, court, threshold=15: 250 <= feet[0][1] <= 700)
    mod.visualize_video_estimated(inferencer, 'clip_9.mp4', csv_output_dir=str(tmp_path), cap=FakeCapture(len(frames)),
                                  court_corners=COURT, start_frame_number=50, draw=False)
    read = lambda name: pd.read_csv(tmp_path / f'clip_9_{name}.csv')
    return read('top'), read('bottom'), read('top_bbox'), read('bottom_bbox')


def flat(p):
    return [v for kp in p['keypoints'] for v in kp]


UMPIRE, LINESMAN = person(150, 800), person(1100, 900)


@pytest.mark.ex('1.7b')
def test_people_outside_the_court_are_ignored(tmp_path, monkeypatch):
    frames = [[UMPIRE, person(640, 650)],   # only one person inside the court
              [UMPIRE, LINESMAN],           # nobody inside
              []]
    top, bottom, _, _ = run_pose(tmp_path, monkeypatch, frames)
    assert top['frame'].tolist() == bottom['frame'].tolist() == [50, 51, 52]
    for df, who in ((top, 'top'), (bottom, 'bottom')):
        assert (df.iloc[:, 1:].to_numpy() == 0).all(), \
            f'{who} player: frames with fewer than two people INSIDE the court must be all zeros'


@pytest.mark.ex('1.7c')
def test_top_and_bottom_player(tmp_path, monkeypatch):
    a_top, a_bottom = person(600, 300), person(640, 650)
    b_top, b_bottom = person(500, 320), person(700, 600)
    frames = [[a_top, a_bottom],   # listed top first
              [b_bottom, b_top]]   # listed bottom first
    top, bottom, top_bbox, bottom_bbox = run_pose(tmp_path, monkeypatch, frames)
    np.testing.assert_allclose(bottom.iloc[:, 1:].to_numpy(), [flat(a_bottom), flat(b_bottom)],
                               err_msg='bottom player = the one with the LARGEST heel y')
    np.testing.assert_allclose(top.iloc[:, 1:].to_numpy(), [flat(a_top), flat(b_top)],
                               err_msg='top player = the other one')
    np.testing.assert_allclose(bottom_bbox[['x1', 'y1', 'x2', 'y2']].to_numpy(),
                               [a_bottom['bbox'][0], b_bottom['bbox'][0]])
    np.testing.assert_allclose(top_bbox[['x1', 'y1', 'x2', 'y2']].to_numpy(), [a_top['bbox'][0], b_top['bbox'][0]])


@pytest.mark.ex('1.7b', '1.7c')
def test_players_among_other_people(tmp_path, monkeypatch):
    p_top, p_bottom = person(600, 300), person(640, 650)
    frames = [[UMPIRE, p_top, p_bottom],
              [p_bottom, LINESMAN, p_top, UMPIRE],
              [p_bottom, UMPIRE]]
    top, bottom, _, _ = run_pose(tmp_path, monkeypatch, frames)
    np.testing.assert_allclose(bottom.iloc[:2, 1:].to_numpy(), [flat(p_bottom)] * 2)
    np.testing.assert_allclose(top.iloc[:2, 1:].to_numpy(), [flat(p_top)] * 2)
    assert (top.iloc[2, 1:] == 0).all() and (bottom.iloc[2, 1:] == 0).all()
