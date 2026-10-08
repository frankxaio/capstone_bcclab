"""Process III - the TemPose stroke classifier (Ex 3.4 - 3.7)."""
import math

import numpy as np
import pytest

from _helpers import (add_to_path, assert_close, check_stages, det_tensor, det_uniform, fill_params,
                      import_repo, load_fixture, trace)


@pytest.fixture(scope='module')
def fx():
    return load_fixture('tempose')


def nn_models():
    return import_repo('TemPose.NN_models', 'torch', 'einops')


def attention_mask():
    import torch
    mask = torch.ones(2, 1, 1, 5)
    mask[0, ..., 3:] = 0            # sample 0: the last two positions are padding
    return mask


# --------------------------------------------------------------------------- Ex 3.4
@pytest.mark.ex('3.4a')
def test_attention(fx):
    m = nn_models()
    import torch
    att = fill_params(m.Attention(8, heads=2, dim_head=4), seed=31).eval()
    x = det_tensor((2, 5, 8), seed=310)
    with torch.no_grad():
        assert_close(att(x), fx['attn/out'], 'Attention output (no mask)')
        assert_close(att(x, attention_mask()), fx['attn/out_mask'], 'Attention output with a padding mask')


@pytest.mark.ex('3.4a')
def test_attention_ignores_masked_positions():
    m = nn_models()
    import torch
    att = fill_params(m.Attention(8, heads=2, dim_head=4), seed=31).eval()
    x = det_tensor((2, 5, 8), seed=310)
    x2 = x.clone()
    x2[0, 3:] += 5.0                # change only the padded positions of sample 0
    with torch.no_grad():
        a, b = att(x, attention_mask()), att(x2, attention_mask())
    assert torch.allclose(a[0, :3], b[0, :3], atol=1e-5), \
        'the output at valid positions must not depend on masked (padding) positions'


@pytest.mark.ex('3.4b')
def test_transformer(fx):
    m = nn_models()
    import torch
    tf = fill_params(m.Transformer(8, depth=2, heads=2, dim_head=4, mlp_dim=16), seed=32).eval()
    x = det_tensor((2, 5, 8), seed=320)
    with torch.no_grad():
        assert_close(tf(x), fx['tf/out'], 'Transformer output (no mask)')
        assert_close(tf(x, attention_mask()), fx['tf/out_mask'], 'Transformer output with a padding mask '
                     '(x = attn(x, mask) + x, then x = ff(x) + x, for every layer)')


# --------------------------------------------------------------------------- Ex 3.5
TEMPOSE_STAGES = [('temporal_transformer', 'in'), ('temporal_transformer', 'mask'),
                  ('interaction_transformer', 'in'), ('mlp_head', 'in')]
TEMPOSE_HINTS = {
    'temporal_transformer': 'Steps 1-3: CLS token prepended to every stream, temporal_embedding[:, :, :t+1] added, '
                            'streams flattened to ((b n), t+1, dim); mask ((b n), 1, 1, t+1) with positions > L set to 0.',
    ('temporal_transformer', 'mask'): 'Step 3: mask of shape ((b n), 1, 1, t+1) - all ones without t_pad; with t_pad, '
                                      'each value repeated n times and, for L valid frames, positions 0..L = 1, L+1..t = 0.',
    'interaction_transformer': 'Steps 4-5: keep the CLS output of every stream (b, n, dim), prepend the interaction '
                               'token and add interaction_embedding[:, :n+1] (check also Ex 3.4a/3.4b).',
    'mlp_head': 'Step 6: CLS token (pool = "cls") or mean over the n+1 tokens (pool = "mean").',
}


def tempose_model(monkeypatch):
    m = import_repo('TemPose.TemPoseII', 'torch', 'einops')
    # the constructor uses get_2d_sincos_pos_embed (Ex 3.6b); fill_params overwrites the result anyway
    monkeypatch.setattr(m, 'get_2d_sincos_pos_embed',
                        lambda dim, n, cls_token=False: np.zeros((n + int(cls_token), dim)))
    model = m.TemPoseII_TF(poses_numbers=6, time_steps=5, num_people=2, num_classes=4, dim=8, kernel_size=3,
                           depth=1, depth_int=1, heads=2, scale_dim=2, dim_head=4, dropout=0.0, emb_dropout=0.0)
    return fill_params(model, seed=35).eval()


def tempose_inputs():
    import torch
    return det_tensor((2, 4, 2, 6), seed=350), det_tensor((2, 4, 6), seed=351), torch.tensor([4, 2])


@pytest.mark.ex('3.5a')
@pytest.mark.parametrize('case', ['pad', 'nopad', 'mean'])
def test_tempose_forward(fx, monkeypatch, case):
    model = tempose_model(monkeypatch)
    import torch
    x, sp, t_pad = tempose_inputs()
    if case == 'mean':
        model.pool = 'mean'
    with torch.no_grad(), trace(model, ['temporal_transformer', 'interaction_transformer', 'mlp_head']) as rec:
        out = model(x, sp, None if case == 'nopad' else t_pad)
    assert tuple(out.shape) == (2, 4), f'expected logits of shape (b, num_classes) = (2, 4), got {tuple(out.shape)}'
    if not np.allclose(out.numpy(), fx[f'tp/{case}/out'], rtol=1e-4, atol=1e-5):
        where = check_stages(rec, fx, f'tp/{case}', TEMPOSE_STAGES, TEMPOSE_HINTS)
        assert_close(out, fx[f'tp/{case}/out'], 'TemPose logits' + (f' -- first difference: {where}' if where else ''))


@pytest.mark.ex('3.5b')
def test_tempose_predict(monkeypatch):
    model = tempose_model(monkeypatch)
    import torch
    logits = torch.tensor([[0.1, 2.0, -1.0, 0.5], [3.0, 0.0, 0.2, 2.9]])
    model.forward = lambda x, sp, t_pad=None: logits            # isolate predict() from forward()
    pred = model.predict(*tempose_inputs())
    assert torch.is_tensor(pred) and pred.device.type == 'cpu', 'predict must return a CPU tensor'
    assert pred.tolist() == [1, 0]


# --------------------------------------------------------------------------- Ex 3.6
def utility():
    return import_repo('TemPose.utility_tempose', 'torch')


def lr_after(e, warm_e=10, num_epochs=110, scale=None):
    u = utility()
    import torch
    group = {'params': [torch.nn.Parameter(torch.zeros(1))]}
    if scale is not None:
        group['lr_scale'] = scale
    opt = torch.optim.SGD([group], lr=123.0)
    u.adjust_lr(opt, 1.0, 0.1, e, num_epochs, warm_e)
    return opt.param_groups[0]['lr']


@pytest.mark.ex('3.6a')
@pytest.mark.parametrize('epoch, expected', [
    (0, 0.0), (5, 0.5), (9, 0.9),                                  # linear warm-up to LR_max = 1
    (10, 1.0), (60, 0.55), (110, 0.1),                             # cosine from 1 down to LR_min = 0.1
    (35, 0.1 + 0.45 * (1 + math.cos(math.pi * 0.25))),
])
def test_learning_rate_schedule(epoch, expected):
    assert lr_after(epoch) == pytest.approx(expected, abs=1e-9)


@pytest.mark.ex('3.6a')
def test_learning_rate_scale():
    assert lr_after(5, scale=0.5) == pytest.approx(0.25)


@pytest.mark.ex('3.6b')
def test_sincos_embedding_small():
    u = utility()
    emb = u.get_1d_sincos_pos_embed_from_grid(4, np.array([0.0, 1.0, 2.0]))
    expected = [[0, 0, 1, 1],
                [np.sin(1), np.sin(0.01), np.cos(1), np.cos(0.01)],
                [np.sin(2), np.sin(0.02), np.cos(2), np.cos(0.02)]]
    np.testing.assert_allclose(emb, expected, rtol=1e-6, atol=1e-7,
                               err_msg='omega = [1, 1/100] for embed_dim = 4; emb = [sin(p*omega) | cos(p*omega)]')


@pytest.mark.ex('3.6b')
def test_sincos_embedding_with_cls_token():
    u = utility()
    emb = u.get_2d_sincos_pos_embed(8, 5, cls_token=True)
    assert emb.shape == (6, 8)
    np.testing.assert_allclose(emb[0], 0)                                    # the CLS row
    np.testing.assert_allclose(emb[1], [0, 0, 0, 0, 1, 1, 1, 1], atol=1e-7)  # position 0
    omega = 1.0 / 10000 ** (np.arange(4) / 4)
    np.testing.assert_allclose(emb[5, :4], np.sin(4 * omega), rtol=1e-6)    # position 4


# --------------------------------------------------------------------------- Ex 3.7
def tiny_tempose():
    import torch

    class TinyTemPose(torch.nn.Module):
        """Stand-in with the same interface as TemPoseII_TF: forward(x, sp, pad) and predict()."""

        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3 * 2 * 4 + 3 * 6, 3)

        def forward(self, x, sp, pad=None):
            return self.lin(torch.cat([x.flatten(1), sp.flatten(1)], dim=1))

        def predict(self, x, sp, pad=None):
            return self.forward(x, sp, pad).argmax(1).cpu()

    return fill_params(TinyTemPose(), seed=37)


def tempose_batches():
    import torch
    batches = []
    for k in range(3):
        jnb = det_tensor((4, 3, 2, 2, 2), seed=370 + k)       # (b, t, n, j, d)
        pos = det_tensor((4, 3, 2, 2), seed=380 + k)          # (b, t, n, 2)
        shuttle = det_tensor((4, 3, 2), seed=390 + k)         # (b, t, 2)
        y = torch.as_tensor((det_uniform((4,), seed=400 + k) + 1) * 1.5, dtype=torch.long).clamp(0, 2)
        batches.append(((jnb, pos, shuttle), torch.full((4,), 3), y))
    return batches


@pytest.mark.ex('3.7')
def test_tempose_training_step(tmp_path, monkeypatch, fx):
    add_to_path('TemPose')                    # train.py does `from utility_tempose import adjust_lr`
    mod = import_repo('TemPose.train', 'torch', 'sklearn')
    import torch
    monkeypatch.chdir(tmp_path)               # train() saves model.pt
    monkeypatch.setattr(mod, 'adjust_lr', lambda *args, **kwargs: None)   # keep lr fixed (Ex 3.6a is tested separately)
    cfg = {'training': {'warm_up': 0, 'epochs': 3, 'learning_rate_start': 0.05, 'learning_rate_min': 0.001},
           'hyperparameters': {'criterion': 'CE', 'optimizer': 'Adam'}}
    model = tiny_tempose()
    mod.train(model, tempose_batches(), 'train', torch.device('cpu'), cfg)
    for name, value in model.state_dict().items():
        assert_close(value, fx[f'train/{name}'], f'parameter {name} after 3 epochs of training '
                     '(loss = criterion(yPred, y) -> zero_grad -> backward -> step, once per batch)')
    assert (tmp_path / 'model.pt').exists()
