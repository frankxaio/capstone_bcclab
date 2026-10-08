"""Process II - hit detection with HitNet (Ex 2.1 - 2.4)."""
import types

import numpy as np
import pytest

from _helpers import (assert_close, check_stages, det_tensor, det_uniform, fill_params,
                      import_repo, load_fixture, require, trace)


@pytest.fixture(scope='module')
def fx():
    return load_fixture('hitnet')


# --------------------------------------------------------------------------- Ex 2.1
N_FRAMES, POSE_DIM = 12, 4


def run_process(monkeypatch, hit=None):
    mod = import_repo('HitNet.data_preparation', 'skimage', 'pandas')
    import pandas as pd
    t = np.arange(N_FRAMES, dtype=float)[:, None]
    d = np.arange(POSE_DIM, dtype=float)[None, :]
    data = {'corners': np.zeros(8),
            'trajectory': types.SimpleNamespace(X=list(100.0 + t[:, 0]), Y=list(200.0 + t[:, 0])),
            'bottom_player': pd.DataFrame(1000.0 + 10 * t + d),
            'top_player': pd.DataFrame(2000.0 + 10 * t + d)}
    if hit is not None:
        data['hit'] = np.asarray(hit)
    monkeypatch.setattr(mod, 'fetch_data', lambda *args, **kwargs: data)
    return mod.process('videos/sample/clip_1/', 'clip_1', for_train=hit is not None)


@pytest.mark.ex('2.1a')
@pytest.mark.parametrize('hit, expected', [
    ([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]),
    ([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
    ([0] * 12, [0] * 12),
])
def test_label_smoothing(monkeypatch, hit, expected):
    _, y = run_process(monkeypatch, hit=hit)
    assert y is not None, 'for_train=True must return the labels'
    assert np.asarray(y).reshape(-1).tolist() == expected, \
        'the frames right before and after every hit must also be labelled 1 (and only those)'


@pytest.mark.ex('2.1b', '2.1c')
def test_padded_windows_and_features(monkeypatch, fx):
    x, y = run_process(monkeypatch)
    assert y is None
    x = np.asarray(x)
    feat = 2 + 2 * POSE_DIM
    assert x.shape == (N_FRAMES, 7, feat), \
        f'expected one (window_size, 2 + 2 * pose_dim) = (7, {feat}) matrix per frame, got {x.shape}'
    for t in (0, 1, 5, 10, 11):
        assert_close(x[t], fx['process/x'][t],
                     f'features of frame {t} (columns: shuttle X, Y | bottom pose | top pose; '
                     f'windows t-3..t+3, edge rows repeated)', rtol=0, atol=1e-6)
    assert_close(x, fx['process/x'], 'all feature windows', rtol=0, atol=1e-6)


# --------------------------------------------------------------------------- Ex 2.2
HITNET_STAGES = [('bigru', 'in'), ('attention', 'in'), ('fc1', 'in')]
HITNET_HINTS = {
    'bigru': 'conv1 -> ReLU -> conv2 -> ReLU -> batchnorm on (B, F, T), then transpose back to (B, T, 128).',
    'attention': 'query / key / value = query_layer / key_layer / value_layer applied to the GRU output.',
    'fc1': 'Pooled features = torch.cat([max over time (self.pool), mean over time], dim=1) of (attn_out + query).',
}


@pytest.mark.ex('2.2')
def test_hitnet_forward(fx):
    mod = import_repo('HitNet.hitnet', 'torch')
    import torch
    net = fill_params(mod.HitNet(70, num_consec=7), seed=21).eval()
    x = det_tensor((3, 7, 70), seed=210)
    with torch.no_grad(), trace(net, [s for s, _ in HITNET_STAGES]) as rec:
        out = net(x)
    assert tuple(out.shape) == (3, 1), f'expected logits of shape (B, 1), got {tuple(out.shape)}'
    if not np.allclose(out.numpy(), fx['net/out'], rtol=1e-4, atol=1e-5):
        where = check_stages(rec, fx, 'net', HITNET_STAGES, HITNET_HINTS)
        assert_close(out, fx['net/out'], 'HitNet output' + (f' -- first difference: {where}' if where else ''))


# --------------------------------------------------------------------------- Ex 2.3
def tiny_classifier():
    import torch
    return fill_params(torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1)), seed=23)


def loaders():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    x = det_tensor((32, 5), seed=230)
    y = torch.as_tensor(det_uniform((32,), seed=231) > 0.3, dtype=torch.long)
    return (DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False),
            DataLoader(TensorDataset(x[:16], y[:16]), batch_size=8, shuffle=False))


@pytest.mark.ex('2.3')
@pytest.mark.parametrize('use_amp', [False, True])
def test_training_step(tmp_path, monkeypatch, fx, use_amp):
    mod = import_repo('HitNet.train', 'torch', 'tqdm', 'sklearn')
    import torch
    monkeypatch.chdir(tmp_path)                                    # train() saves best_model.pth
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)  # CPU only: AMP becomes a no-op
    model = tiny_classifier()
    train_loader, val_loader = loaders()
    mod.train(model, train_loader, val_loader, num_epochs=2, learning_rate=1e-2, use_amp=use_amp,
              device=torch.device('cpu'))
    for name, value in model.state_dict().items():
        assert_close(value, fx[f'train/{name}'], f'parameter {name} after 2 epochs of training '
                     '(zero_grad -> forward -> BCE loss -> backward -> optimizer step, once per batch)')


# --------------------------------------------------------------------------- Ex 2.4
LOGITS = {'clip_1': [-3.0, 0.5, 0.3, 2.0], 'clip_2': [0.0, 0.24, 0.25]}   # threshold 0.56 <=> logit 0.2412
START = {'clip_1': 100, 'clip_2': 200}


@pytest.mark.ex('2.4a', '2.4b')
def test_hits_csv(tmp_path, monkeypatch):
    mod = import_repo('HitNet.predict', 'torch', 'pandas')
    import pandas as pd
    import torch

    class FakeHitNet(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x):              # the logit of every sample is stored in x[:, 0, 0]
            return x[:, 0, :1]

    videos = tmp_path / 'videos'
    for clip, logits in LOGITS.items():
        (videos / clip).mkdir(parents=True)
    (videos / 'court.txt').write_text('not a clip')
    weights = tmp_path / 'weights.pth'
    torch.save({}, weights)

    def fake_data(clip_dir, rally):
        x = np.zeros((len(LOGITS[rally]), 7, 70), dtype=np.float32)
        x[:, 0, 0] = LOGITS[rally]
        return x

    monkeypatch.setattr(mod, 'HitNet', FakeHitNet)
    monkeypatch.setattr(mod, 'make_data_for_predict', fake_data)
    monkeypatch.setattr(mod, 'MODEL_PATH', str(weights))
    monkeypatch.setattr(mod, 'batch_size', 2)                      # several batches per clip
    mod.predict(str(videos), dict(START), device=torch.device('cpu'))

    for clip, expected in [('clip_1', [0, 1, 1, 1]), ('clip_2', [0, 0, 1])]:
        df = pd.read_csv(videos / clip / f'{clip}_hits.csv')
        assert list(df.columns) == ['frame', 'hit'], f"{clip}: columns must be ['frame', 'hit'], got {list(df.columns)}"
        assert df['frame'].tolist() == [START[clip] + i for i in range(len(expected))], \
            f'{clip}: frame = start_frame[clip] + row index (absolute frame numbers)'
        assert df['hit'].tolist() == expected, f'{clip}: hit = sigmoid(logit) >= threshold (0.56)'
