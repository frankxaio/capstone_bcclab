"""Keep bulk assets and completed solutions out of the student ZIP."""
import zipfile

import pytest

from scripts.package_student import ROOT, build_archive


def test_student_archive_is_reproducible_and_contains_exercises(tmp_path):
    if 'raise NotImplementedError("TODO [Ex' not in (ROOT / 'main.py').read_text():
        with pytest.raises(ValueError, match='expected 43 unfinished'):
            build_archive(tmp_path / 'student.zip')
        return
    first = build_archive(tmp_path / 'first.zip')
    second = build_archive(tmp_path / 'second.zip')
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        prefix = 'BadmintonAI-End2End-exercises/'
        assert prefix + 'tests/fixtures/tempose.npz' in names
        assert prefix + 'scripts/weights.json' in names
        assert prefix + 'court_detection/LICENSE' in names
        assert prefix + 'ai_badminton/LICENCE.txt' in names
        assert not any(name.endswith(('.pt', '.pth', '.mp4', '.pyc')) for name in names)
        assert not any('/build/' in name or '/__pycache__/' in name for name in names)
        assert b'raise NotImplementedError("TODO [Ex 3.9]")' in archive.read(prefix + 'main.py')


def test_completed_copy_cannot_be_packaged_as_starter(tmp_path):
    (tmp_path / 'main.py').write_text('print("completed solution")\n')
    with pytest.raises(ValueError, match='expected 43 unfinished'):
        build_archive(tmp_path / 'student.zip', root=tmp_path)
