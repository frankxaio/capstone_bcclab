"""Process II - cut a fixed window around every hit for TemPose (Ex 2.5)."""
import numpy as np
import pytest

from _helpers import import_repo, require


def windows_module():
    return import_repo('TemPose.predict_by_hit', 'torch')


# --------------------------------------------------------------------------- Ex 2.5a
ARR = np.arange(20).reshape(10, 2)


@pytest.mark.ex('2.5a')
@pytest.mark.parametrize('center, T, rows, pad', [
    (5, 4, (3, 7), 0),     # rows 3..6
    (5, 6, (2, 8), 0),
    (1, 6, (0, 4), 2),     # clipped at the start -> zero rows appended at the END
    (0, 4, (0, 2), 2),
    (8, 6, (5, 10), 1),    # clipped at the end
    (9, 4, (7, 10), 1),
])
def test_slice_and_pad(center, T, rows, pad):
    mod = windows_module()
    out = mod.slice_and_pad(ARR, center, T)
    expected = np.concatenate([ARR[rows[0]:rows[1]], np.zeros((pad, 2), dtype=ARR.dtype)])
    assert out.shape == (T, 2), f'expected shape ({T}, 2), got {out.shape}'
    np.testing.assert_array_equal(out, expected)


@pytest.mark.ex('2.5a')
def test_slice_and_pad_keeps_shape_and_dtype():
    mod = windows_module()
    arr = np.ones((10, 2, 36, 2), dtype=np.float32)
    out = mod.slice_and_pad(arr, 8, 6)
    assert out.shape == (6, 2, 36, 2) and out.dtype == np.float32
    assert (out[:5] == 1).all() and (out[5:] == 0).all()


# --------------------------------------------------------------------------- Ex 2.5b
N, T = 40, 10
HITS = [101, 120, 137]      # includes windows clipped at both ends of the clip


def encoded_inputs():
    t = np.arange(N)[:, None, None, None]
    m = np.arange(2)[None, :, None, None]
    j = np.arange(36)[None, None, :, None]
    d = np.arange(2)[None, None, None, :]
    jnb = (t * 1000 + m * 100 + j * 2 + d).astype(np.float32)[None]              # (1, N, 2, 36, 2)
    pos = (np.arange(N)[:, None, None] * 10 + np.arange(2)[None, :, None] * 2
           + np.arange(2)[None, None, :] + 0.5).astype(np.float32)[None]          # (1, N, 2, 2)
    shuttle = (np.arange(N)[:, None] * 10 + np.arange(2)[None, :] + 0.25).astype(np.float32)[None]  # (1, N, 2)
    return jnb, pos, shuttle


@pytest.mark.ex('2.5b')
def test_per_hit_model_inputs(tmp_path):
    mod = windows_module()
    require('pandas')
    import pandas as pd
    import torch
    for name, arr in zip(['JnB_bone', 'pos', 'shuttle'], encoded_inputs()):
        np.save(tmp_path / f'{name}.npy', arr)
    pd.DataFrame({'frame': np.arange(100, 100 + N), 'hit': [int(f in HITS) for f in range(100, 100 + N)]}) \
        .to_csv(tmp_path / 'clip_1_hits.csv', index=False)

    class FakeTemPose:
        def __init__(self):
            self.inputs = []

        def predict(self, x, sp, pad=None):
            self.inputs.append((x.detach().cpu().numpy(), sp.detach().cpu().numpy(), pad.item()))
            return torch.tensor([[0, 17, 34][len(self.inputs) - 1]])

    model = FakeTemPose()
    frames, strokes = mod.per_hit_predict(model, tmp_path, tmp_path / 'clip_1_hits.csv',
                                          {'model': {'sequence_length': T}}, torch.device('cpu'))
    names = import_repo('TemPose.data.dataset').get_stroke_types()
    assert list(frames) == HITS
    assert list(strokes) == [names[0], names[17], names[34]]

    for (x, sp, valid_len), hit in zip(model.inputs, HITS):
        start = max(0, hit - 100 - T // 2)
        end = min(N, hit - 100 + T // 2)
        assert valid_len == end - start
        assert x.shape == (1, T, 2, 72), f'x must be (1, T, 2, (J+B)*2) = (1, {T}, 2, 72), got {x.shape}'
        assert sp.shape == (1, T, 6), f'sp must be (1, T, 6), got {sp.shape}'
        for k in range(T):
            frame = start + k
            if frame >= end:                    # padding follows the clipped window
                assert (x[0, k] == 0).all() and (sp[0, k] == 0).all()
                continue
            expected_x = frame * 1000 + np.arange(2)[:, None] * 100 + np.arange(72)[None, :]
            np.testing.assert_array_equal(x[0, k], expected_x,
                                          err_msg=f'hit {hit}, window row {k}: x[0, k, m] must hold the 36 (x, y) '
                                                  f'pairs of player m flattened as [x0, y0, x1, y1, ...]')
            np.testing.assert_allclose(sp[0, k], frame * 10 + np.array([0, 1, 2, 3, 0, 1]) + np.array([.5] * 4 + [.25] * 2),
                                       err_msg=f'hit {hit}, window row {k}: sp = [top x, top y, bottom x, '
                                               f'bottom y, shuttle x, shuttle y]')
