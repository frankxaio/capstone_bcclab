"""Process I - shuttlecock tracking with TrackNet (Ex 1.4, 1.5, 1.6)."""
import numpy as np
import pytest

from _helpers import (assert_close, check_stages, det_tensor, det_uniform, fill_params,
                      get_fingerprint, import_repo, load_fixture, matches_fingerprint, require,
                      trace, write_video)

TRACKNET_STAGES = [('bottleneck', 'in'), ('cbam0_2', 'in'), ('up_block_1', 'in'),
                   ('up_block_2', 'in'), ('up_block_3', 'in'), ('predictor', 'in')]
TRACKNET_HINTS = {
    'bottleneck': 'Check the encoder: down_block_1..3, each followed by nn.MaxPool2d((2, 2), stride=(2, 2)).',
    'cbam0_2': 'x3 is the output of down_block_3 (before its max pooling).',
    'up_block_1': 'Expected torch.cat([nn.Upsample(scale_factor=2)(x), cbam0_2(x3)], dim=1) - upsampled x '
                  'FIRST, nearest upsampling (and Ex 1.4a must pass: CBAM uses the channel attention).',
    'up_block_2': 'After up_block_1 apply cbam1, then torch.cat([upsampled x, cbam1_2(x2)], dim=1) - upsampled x FIRST.',
    'up_block_3': 'After up_block_2 apply cbam2, then torch.cat([upsampled x, cbam2_2(x1)], dim=1) - upsampled x FIRST.',
    'predictor': 'After up_block_3 apply cbam3, then the 1x1 predictor and a sigmoid.',
}


@pytest.fixture(scope='module')
def fx():
    return load_fixture('tracknet')


def model_module():
    return import_repo('TrackNetV3.model', 'torch')


# --------------------------------------------------------------------------- Ex 1.4
def channel_attention_input():
    # strong per-channel offsets, so that both the average- and the max-pooling branch matter
    return det_tensor((2, 32, 5, 7), seed=110) + 2 * det_tensor((1, 32, 1, 1), seed=111)


@pytest.mark.ex('1.4a')
def test_channel_attention(fx):
    m = model_module()
    import torch
    net = fill_params(m.ChannelAttentionModule(32), seed=11).eval()
    with torch.no_grad():
        out = net(channel_attention_input())
    assert tuple(out.shape) == (2, 32, 1, 1), f'expected weights of shape (B, C, 1, 1), got {tuple(out.shape)}'
    assert_close(out, fx['ca/out'], 'ChannelAttentionModule output')


@pytest.mark.ex('1.4b')
def test_spatial_attention(fx):
    m = model_module()
    import torch
    net = fill_params(m.SpatialAttentionModule(), seed=12).eval()
    with torch.no_grad():
        out = net(det_tensor((2, 8, 5, 7), seed=120))
    assert tuple(out.shape) == (2, 1, 5, 7), f'expected weights of shape (B, 1, H, W), got {tuple(out.shape)}'
    assert_close(out, fx['sa/out'], 'SpatialAttentionModule output')


@pytest.mark.ex('1.4c')
def test_tracknet_forward(fx):
    m = model_module()
    import torch
    net = fill_params(m.TrackNetV2(in_dim=9, out_dim=3), seed=13).eval()
    x = (det_tensor((1, 9, 16, 32), seed=130) + 1) / 2
    with torch.no_grad(), trace(net, [s for s, _ in TRACKNET_STAGES]) as rec:
        out = net(x)
    assert tuple(out.shape) == (1, 3, 16, 32), f'expected (B, F, H, W) = (1, 3, 16, 32), got {tuple(out.shape)}'
    if not np.allclose(out.numpy(), fx['tn/out'], rtol=1e-4, atol=1e-5):
        where = check_stages(rec, fx, 'tn', TRACKNET_STAGES, TRACKNET_HINTS)
        assert_close(out, fx['tn/out'], 'TrackNetV2 output' + (f' -- first difference: {where}' if where else ''))


# --------------------------------------------------------------------------- Ex 1.5
def tn_utils():
    return import_repo('TrackNetV3.utils', 'torch', 'cv2')


@pytest.mark.ex('1.5a')
def test_frame_unit(fx):
    u = tn_utils()
    ys, xs = np.mgrid[0:36, 0:64]
    frames = [np.stack([(xs * 3 + ys * 5 + c * 50 + f * 20) % 256 for c in range(3)], axis=-1).astype(np.uint8)
              for f in range(6)]
    out = u.get_frame_unit(frames, 3)
    assert hasattr(out, 'dtype') and str(out.dtype) == 'torch.float32', 'get_frame_unit must return a float32 tensor'
    assert tuple(out.shape) == (2, 9, u.HEIGHT, u.WIDTH), f'expected (2, 9, {u.HEIGHT}, {u.WIDTH}), got {tuple(out.shape)}'
    a = out.numpy()
    assert 0.0 <= a.min() and a.max() <= 1.0, 'values must be scaled to [0, 1]'
    means = a.mean(axis=(2, 3))
    assert_close(means, fx['unit/channel_means'], 'mean of every output channel (channel / frame order?)',
                 rtol=0, atol=2e-3)
    assert matches_fingerprint(a, get_fingerprint(fx, 'unit/out'), rtol=0, atol=2.5 / 255), \
        'resized pixel values differ from the reference (use cv2.resize(img, (WIDTH, HEIGHT)) with the default interpolation)'


def blob_map(*rects, shape=(288, 512)):
    h = np.zeros(shape, dtype=np.uint8)
    for x, y, w, hh in rects:
        h[y:y + hh, x:x + w] = 255
    return h


@pytest.mark.ex('1.5b')
@pytest.mark.parametrize('rects, expected', [
    ([], (0, 0)),
    ([(10, 20, 10, 6)], (15, 23)),
    ([(30, 40, 3, 5)], (31, 42)),                        # odd sizes: int(30 + 1.5), int(40 + 2.5)
    ([(100, 50, 3, 3), (200, 60, 8, 5)], (204, 62)),     # the LARGER blob wins
    ([(300, 200, 9, 9), (20, 30, 4, 4)], (304, 204)),
])
def test_object_center(rects, expected):
    u = tn_utils()
    cx, cy = u.get_object_center(blob_map(*rects))
    assert (cx, cy) == expected
    assert isinstance(cx, (int, np.integer)) and isinstance(cy, (int, np.integer)), 'return ints'


@pytest.mark.ex('1.5c')
def test_weighted_bce_simple_values():
    u = tn_utils()
    import torch
    y = torch.tensor([1.0, 0.0])
    y_pred = torch.tensor([0.5, 0.5])
    assert float(u.WeightedBinaryCrossEntropy(y, y_pred)) == pytest.approx(0.25 * np.log(2), rel=1e-5)
    # a confident wrong prediction must give a large but FINITE loss (clamping)
    loss = float(u.WeightedBinaryCrossEntropy(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])))
    assert np.isfinite(loss), 'loss is not finite - clamp the arguments of log to [1e-7, 1]'
    assert loss == pytest.approx(-np.log(np.float32(1e-7)), rel=1e-4)


@pytest.mark.ex('1.5c')
def test_weighted_bce_matches_reference(fx):
    u = tn_utils()
    import torch
    y = torch.as_tensor(det_uniform((2, 3, 4, 5), seed=150) > 0.4, dtype=torch.float32)
    y_pred = (det_tensor((2, 3, 4, 5), seed=151) + 1) / 2
    y_pred[0, 0, 0, :2] = torch.tensor([0.0, 1.0])
    loss = u.WeightedBinaryCrossEntropy(y, y_pred)
    assert loss.dim() == 0, 'return the mean loss (a scalar tensor)'
    assert float(loss) == pytest.approx(float(fx['wbce/loss']), rel=1e-5)


# --------------------------------------------------------------------------- Ex 1.6
# The model predicts the shuttle at these (x, y) positions of the 512x288 heatmap; frame 4 has none.
SHUTTLE = {0: (20, 10), 1: (22, 11), 2: (24, 12), 3: (26, 13), 5: (30, 15)}


def run_trajectory(tmp_path, monkeypatch, frame_count=6):
    mod = import_repo('TrackNetV3.predict', 'torch', 'cv2', 'tqdm')
    require('pandas')
    import pandas as pd
    import torch
    H, W = mod.HEIGHT, mod.WIDTH
    video = write_video(tmp_path / 'clip_1.mp4', [np.zeros((576, 1024, 3), np.uint8)] * frame_count)   # ratio = 2

    class FakeTrackNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.calls = 0

        def forward(self, x):
            assert tuple(x.shape) == (1, 9, H, W), f'unexpected model input {tuple(x.shape)}'
            out = torch.full((1, 3, H, W), 0.3)              # background below the threshold
            for k in range(3):
                f = 3 * self.calls + k
                if f in SHUTTLE:
                    sx, sy = SHUTTLE[f]
                    out[0, k, sy, sx] = 0.9
                else:
                    out[0, k] = 0.45                          # "almost" a shuttle everywhere
            self.calls += 1
            return out

    heatmaps = []

    def fake_center(heatmap):
        h = np.asarray(heatmap)
        heatmaps.append(h)
        ys, xs = np.nonzero(h)
        return (0, 0) if len(xs) == 0 else (int(xs[0]), int(ys[0]))

    monkeypatch.setattr(mod, 'get_frame_unit', lambda frames, n: torch.zeros(1, 3 * n, H, W))
    monkeypatch.setattr(mod, 'get_object_center', fake_center)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **k: self)   # run the `.cuda()` path on CPU
    csv = mod.predict_traj(video, str(tmp_path), FakeTrackNet(), start_frame_number=100)
    return pd.read_csv(csv), heatmaps, (H, W)


@pytest.mark.ex('1.6a')
def test_heatmaps_are_thresholded(tmp_path, monkeypatch):
    _, heatmaps, (H, W) = run_trajectory(tmp_path, monkeypatch)
    assert len(heatmaps) == 6, f'expected one heatmap per frame (6), got {len(heatmaps)} - reshape to (num_frame * batch_size, HEIGHT, WIDTH)'
    for f, h in enumerate(heatmaps):
        assert h.shape == (H, W), f'frame {f}: heatmap shape {h.shape}, expected {(H, W)}'
        assert h.dtype == np.uint8, f'frame {f}: heatmap dtype {h.dtype}, expected uint8'
        assert set(np.unique(h)) <= {0, 255}, f'frame {f}: heatmap values must be 0 or 255'
        lit = list(zip(*np.nonzero(h)[::-1]))
        assert lit == ([SHUTTLE[f]] if f in SHUTTLE else []), \
            f'frame {f}: pixels > 0.5 must become 255, everything else 0 (lit pixels {lit[:5]})'


@pytest.mark.ex('1.6b')
def test_trajectory_csv(tmp_path, monkeypatch):
    df, _, _ = run_trajectory(tmp_path, monkeypatch)
    assert list(df.columns) == ['Frame', 'Visibility', 'X', 'Y']
    assert df['Frame'].tolist() == [100, 101, 102, 103, 104, 105]
    assert df['Visibility'].tolist() == [1, 1, 1, 1, 0, 1]
    assert df['X'].tolist() == [40, 44, 48, 52, 0, 60], 'X must be scaled back to the video resolution (x * ratio)'
    assert df['Y'].tolist() == [20, 22, 24, 26, 0, 30], 'Y must be scaled back to the video resolution (y * ratio)'


@pytest.mark.ex('1.6b')
@pytest.mark.parametrize('frame_count', [4, 5, 7, 8])
def test_trajectory_tail_has_one_row_per_frame(tmp_path, monkeypatch, frame_count):
    df, _, _ = run_trajectory(tmp_path, monkeypatch, frame_count)
    assert df['Frame'].tolist() == list(range(100, 100 + frame_count))
