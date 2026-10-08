"""Weight verification must distinguish pointers, corruption, and usable files."""
import hashlib

from scripts.fetch_weights import verified


def test_weight_verification(tmp_path):
    path = tmp_path / 'model.pt'
    payload = b'example model bytes'
    item = {'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
    assert not verified(path, item)
    path.write_text('version https://git-lfs.github.com/spec/v1\n')
    assert not verified(path, item)
    path.write_bytes(b'x' * len(payload))
    assert not verified(path, item)
    path.write_bytes(payload)
    assert verified(path, item)
