"""Process I - rally clipping (Ex 1.1, 1.2) and the court sanity check (Ex 1.3)."""
import numpy as np
import pytest

from _helpers import import_repo


# --------------------------------------------------------------------------- Ex 1.1
def signal(*segments):
    """signal((0.1, 20), (0.9, 30)) -> 20 frames at 0.1 followed by 30 frames at 0.9"""
    return np.concatenate([np.full(n, v, dtype=float) for v, n in segments])


def cut_points(a, **kwargs):
    mod = import_repo('rally_clipping.get_result', 'torch')
    out = mod.determine_cut_points(np.asarray(a, dtype=float), **kwargs)
    assert isinstance(out, np.ndarray), f'determine_cut_points must return a numpy array, got {type(out).__name__}'
    return out.tolist()


@pytest.mark.ex('1.1')
def test_single_rally():
    assert cut_points(signal((0.1, 20), (0.9, 30), (0.1, 30))) == [20, 49]


@pytest.mark.ex('1.1')
def test_two_rallies():
    a = signal((0.1, 15), (0.9, 20), (0.1, 25), (0.95, 18), (0.05, 20))
    assert cut_points(a) == [15, 34, 60, 77]


@pytest.mark.ex('1.1')
def test_video_starts_in_the_middle_of_a_rally():
    assert cut_points(signal((0.9, 30), (0.1, 30))) == [3, 29]


@pytest.mark.ex('1.1')
def test_video_ends_in_the_middle_of_a_rally():
    # odd number of cut points -> the last rally is closed with len(a)
    assert cut_points(signal((0.1, 30), (0.9, 30))) == [30, 60]


@pytest.mark.ex('1.1')
def test_slow_changes_are_not_cuts():
    # no jump between neighbouring 3-frame means is >= steep_th
    assert cut_points(np.linspace(0.0, 1.0, 100)) == []
    assert cut_points(np.concatenate([np.linspace(0.2, 0.8, 60), np.full(30, 0.8)])) == []


@pytest.mark.ex('1.1')
def test_threshold_parameter_is_used():
    a = signal((0.1, 20), (0.65, 30), (0.1, 20))
    assert cut_points(a) == [20, 49]
    assert cut_points(a, th=0.7) == []          # the plateau never reaches th


# --------------------------------------------------------------------------- Ex 1.2
def run_infer(batch_logits):
    mod = import_repo('rally_clipping.img_predict', 'torch', 'tqdm')
    import torch

    class FakeCourtViewClassifier(torch.nn.Module):
        def forward(self, x):                      # one logit per frame: (B,)
            return x[:, 0, 0, 0]

    loader = []
    for logits in batch_logits:
        batch = torch.zeros(len(logits), 3, 4, 4)
        batch[:, 0, 0, 0] = torch.tensor(logits, dtype=torch.float32)
        loader.append(batch)
    return mod.infer(FakeCourtViewClassifier(), loader, torch.device('cpu'))


@pytest.mark.ex('1.2')
def test_infer_returns_probabilities_for_every_frame():
    out = run_infer([[0.0, np.log(3.0), -np.log(3.0)], [np.log(4.0), -np.log(9.0)]])
    assert isinstance(out, np.ndarray), f'infer must return a numpy array, got {type(out).__name__}'
    assert out.ndim == 1, f'expected a 1-D array (one probability per frame), got shape {out.shape}'
    np.testing.assert_allclose(out, [0.5, 0.75, 0.25, 0.8, 0.1], rtol=1e-5, atol=1e-6)


@pytest.mark.ex('1.2')
def test_infer_last_batch_with_a_single_frame():
    out = run_infer([[0.0, np.log(3.0)], [-np.log(3.0)]])
    assert out.shape == (3,), f'expected shape (3,), got {out.shape}'
    np.testing.assert_allclose(out, [0.5, 0.75, 0.25], rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- Ex 1.3
HOMOGRAPHY = ('Homography Matrix:\n[0.0258, 0.0088, -13.455;\n 0.00045, 0.1105, -33.94;\n'
              ' 2.05e-05, 0.00289, 1]\n')


def court_ok(tmp_path, TL, BL, BR, TR):
    mod = import_repo('utils')
    path = tmp_path / 'court.txt'
    lines = [f'{x};{y}' for x, y in (TL, BL, BR, TR)] + ['369.383;308.912', '912.41;309.382']
    path.write_text('\n'.join(lines) + '\n' + HOMOGRAPHY)
    return mod.check_court_coordinates(path)


@pytest.mark.ex('1.3')
def test_plausible_court(tmp_path):
    # the real output of the court detector for the sample video
    assert court_ok(tmp_path, (416.684, 305.409), (296.303, 658.885), (985.836, 657.18), (864.554, 303.577)) is True


@pytest.mark.ex('1.3')
@pytest.mark.parametrize('corners, problem', [
    (((900, 305), (296, 658), (985, 657), (864, 303)), 'TL is right of TR'),
    (((416, 305), (990, 658), (985, 657), (864, 303)), 'BL is right of BR'),
    (((416, 700), (296, 658), (985, 657), (864, 303)), 'TL is below BL'),
    (((416, 305), (296, 658), (985, 657), (864, 690)), 'TR is below BR'),
    (((864, 658), (985, 305), (416, 303), (296, 657)), 'corners are mirrored'),
])
def test_implausible_courts(tmp_path, corners, problem):
    assert court_ok(tmp_path, *corners) is False, problem


@pytest.mark.ex('1.3')
def test_equal_coordinates_are_accepted(tmp_path):
    # TL_x == TR_x and TR_y == BR_y: "not to the right / not below" -> still accepted
    assert court_ok(tmp_path, (600, 300), (300, 650), (980, 650), (600, 650)) is True
