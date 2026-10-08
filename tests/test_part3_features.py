"""Process III - features for TemPose: court coordinates, normalisation, bones (Ex 3.1 - 3.3)."""
import numpy as np
import pytest

from _helpers import add_to_path, import_repo, require


def tempose_utils():
    return import_repo('TemPose.utils')


# --------------------------------------------------------------------------- Ex 3.1
AFFINE = np.array([[2.0, 0.0, 1.0], [0.0, 3.0, 2.0], [0.0, 0.0, 1.0]])     # (x, y) -> (2x + 1, 3y + 2)


@pytest.mark.ex('3.1a')
def test_project_affine():
    u = tempose_utils()
    P = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    np.testing.assert_allclose(u.project(AFFINE, P), [[1, 3, 5], [2, 5, 5]])


@pytest.mark.ex('3.1a')
def test_project_divides_by_w():
    u = tempose_utils()
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])      # w = x + 1
    P = np.array([[1.0, 3.0], [2.0, 6.0], [1.0, 1.0]])
    np.testing.assert_allclose(u.project(H, P), [[0.5, 0.75], [1.0, 1.5]])
    # homogeneous input whose last row is not 1
    np.testing.assert_allclose(u.project(np.eye(3), np.array([[2.0], [4.0], [2.0]])), [[1.0], [2.0]])


@pytest.mark.ex('3.1b')
@pytest.mark.parametrize('points', [2, 3])
def test_to_court_coordinate(points):
    u = tempose_utils()
    cam = np.arange(2 * points * 2, dtype=float).reshape(2, points, 2) + 1      # (T=2, P, 2)
    out = u.to_court_coordinate(cam, AFFINE)
    assert out.shape == (2, points, 2), f'expected (T, P, 2) = (2, {points}, 2), got {out.shape}'
    expected = np.stack([2 * cam[..., 0] + 1, 3 * cam[..., 1] + 2], axis=-1)
    np.testing.assert_allclose(out, expected, err_msg='every (x, y) point must be projected on its own')


# --------------------------------------------------------------------------- Ex 3.2
# frame 0: bbox (0, 0)-(3, 4) -> diagonal 5;  frame 1: bbox (10, 20)-(16, 28) -> diagonal 10
JOINTS = np.array([[[3.0, 4.0], [0.0, 8.0], [6.0, 0.0]],
                   [[13.0, 24.0], [0.0, 0.0], [16.0, 28.0]]])
BBOX = np.array([[0.0, 0.0, 3.0, 4.0], [10.0, 20.0, 16.0, 28.0]])
NORM = np.array([[[0.6, 0.8], [0.0, 1.6], [1.2, 0.0]],
                 [[0.3, 0.4], [0.0, 0.0], [0.6, 0.8]]])
CENTERED = np.array([[[0.3, 0.4], [-0.3, 1.2], [0.9, -0.4]],
                     [[0.0, 0.0], [-0.3, -0.4], [0.3, 0.4]]])


@pytest.mark.ex('3.2a')
def test_normalize_joints():
    u = tempose_utils()
    np.testing.assert_allclose(u.normalize_joints(JOINTS, BBOX, center_align=False), NORM, atol=1e-12,
                               err_msg='(joint - bbox top-left) / bbox diagonal; coordinates that are 0 stay 0')
    np.testing.assert_allclose(u.normalize_joints(JOINTS, BBOX, center_align=True), CENTERED, atol=1e-12,
                               err_msg='center_align: subtract the normalised bbox centre from every joint')
    np.testing.assert_allclose(u.normalize_joints(JOINTS, BBOX), CENTERED, atol=1e-12,
                               err_msg='center_align defaults to True here')


@pytest.mark.ex('3.2a')
def test_normalize_joints_with_video_height():
    u = tempose_utils()
    out = u.normalize_joints(JOINTS, BBOX, v_height=20, center_align=False)       # dist = 20 / 4 = 5
    np.testing.assert_allclose(out[1], [[0.6, 0.8], [0.0, 0.0], [1.2, 1.6]], atol=1e-12)


COURT = {'H': np.eye(3), 'border_L': 0.0, 'border_R': 4.0, 'border_U': 0.0, 'border_D': 8.0}


@pytest.mark.ex('3.2b')
def test_normalize_position():
    u = tempose_utils()
    heels = np.array([[[0.0, 0.0], [2.0, 4.0]],       # mean (1, 2)
                      [[4.0, 8.0], [4.0, 8.0]],       # mean (4, 8): bottom-right corner
                      [[2.0, 4.0], [2.0, 4.0]]])      # court centre
    out = u.normalize_position(heels, COURT)
    assert out.shape == (3, 1, 2), f'expected (T, 1, 2), got {out.shape}'
    np.testing.assert_allclose(out[:, 0], [[0.25, 0.25], [1.0, 1.0], [0.5, 0.5]])


def prepare_train():
    add_to_path('TemPose')                  # prepare_train.py imports `data.dataset` like a script
    return import_repo('TemPose.prepare_train', 'torch')


@pytest.mark.ex('3.2c')
def test_normalize_position_training_layout():
    mod = prepare_train()
    arr = np.array([[1.0, 4.0, 0.0, 2.0], [2.0, 8.0, 0.0, 4.0]])                 # (2, N): x row, y row
    np.testing.assert_allclose(mod.normalize_position(arr, COURT), [[0.25, 1.0, 0.0, 0.5], [0.25, 1.0, 0.0, 0.5]])


@pytest.mark.ex('3.2d')
@pytest.mark.parametrize('module', ['TemPose.prepare_train', 'MMPose.detect_pose'])
def test_normalize_joints_training_layout(module):
    mod = prepare_train() if module == 'TemPose.prepare_train' else import_repo(module)
    # (m, J, 2) with m = 2 players - same numbers as Ex 3.2a
    np.testing.assert_allclose(mod.normalize_joints(JOINTS, BBOX), NORM, atol=1e-12,
                               err_msg=f'{module}: center_align defaults to False here')
    np.testing.assert_allclose(mod.normalize_joints(JOINTS, BBOX, center_align=True), CENTERED, atol=1e-12)
    np.testing.assert_allclose(mod.normalize_joints(JOINTS, BBOX, v_height=20)[1],
                               [[0.6, 0.8], [0.0, 0.0], [1.2, 1.6]], atol=1e-12)


# --------------------------------------------------------------------------- Ex 3.3
@pytest.mark.ex('3.3a')
def test_compute_bones():
    mod = import_repo('TemPose.generate_npy')
    kps = np.array([[[0.0, 0.0], [1.0, 2.0], [4.0, 6.0]],
                    [[1.0, 1.0], [1.0, 1.0], [0.0, 5.0]]])
    out = mod.compute_bones(kps, [(0, 1), (1, 2), (2, 0)])
    assert out.shape == (2, 3, 2), f'expected (T, B, 2), got {out.shape}'
    np.testing.assert_allclose(out, [[[1, 2], [3, 4], [-4, -6]],
                                     [[0, 0], [-1, 4], [1, -4]]], err_msg='bone (i, j) = joint j - joint i')


@pytest.mark.ex('3.3b')
def test_joints_and_bones_tensor(tmp_path, monkeypatch):
    mod = import_repo('TemPose.generate_npy', 'pandas')
    import pandas as pd
    T = 5
    t = np.arange(T, dtype=float)[:, None, None]
    j = np.arange(17, dtype=float)[None, :, None]
    xy = np.array([0.25, 0.5])[None, None, :]
    top, bottom = t * 100 + j + xy, t * 100 + j + xy + 50                            # (T, 17, 2)
    for name, kps in (('top', top), ('bottom', bottom)):
        cols = ['frame'] + [f'x{i}' for i in range(17)] + [f'y{i}' for i in range(17)]
        pd.DataFrame(np.column_stack([np.arange(T), kps.reshape(T, -1)]), columns=cols) \
            .to_csv(tmp_path / f'clip_1_{name}.csv', index=False)
        pd.DataFrame({'frame': range(T), 'x1': 0.0, 'y1': 0.0, 'x2': 10.0, 'y2': 20.0}) \
            .to_csv(tmp_path / f'clip_1_{name}_bbox.csv', index=False)
    pd.DataFrame({'Frame': range(T), 'Visibility': 1, 'X': 640, 'Y': 360}).to_csv(tmp_path / 'clip_1_ball.csv', index=False)

    # isolate Ex 3.3b from the other exercises used by process_clip
    monkeypatch.setattr(mod, 'get_court_info', lambda txt: dict(COURT))
    monkeypatch.setattr(mod, 'to_court_coordinate', lambda arr, H: arr)
    monkeypatch.setattr(mod, 'normalize_position', lambda arr, info: np.zeros((arr.shape[0], 1, 2)))
    monkeypatch.setattr(mod, 'normalize_joints', lambda arr, bbox, *a, **k: np.asarray(arr, dtype=float))
    marker = lambda kps, bones: np.repeat(kps[:, :1, :] + 1000.0, len(bones), axis=1)   # fake bones
    monkeypatch.setattr(mod, 'compute_bones', marker)

    mod.process_clip('clip_1', str(tmp_path), 'court.txt')
    hp = np.load(tmp_path / 'JnB_bone.npy')
    assert hp.shape == (1, T, 2, 17 + 19, 2), f'expected (1, T, 2, J+B, 2) = (1, {T}, 2, 36, 2), got {hp.shape}'
    for m, kps, who in ((0, top, 'top'), (1, bottom, 'bottom')):
        np.testing.assert_allclose(hp[0, :, m, :17], kps, err_msg=f'human_pose[0, :, {m}, :17] must be the {who} joints')
        np.testing.assert_allclose(hp[0, :, m, 17:], marker(kps, range(19)),
                                   err_msg=f'human_pose[0, :, {m}, 17:] must be the {who} bones (compute_bones(..., COCO_BONES))')
    assert np.load(tmp_path / 'pos.npy').shape == (1, T, 2, 2)
    assert np.load(tmp_path / 'shuttle.npy').shape == (1, T, 2)
