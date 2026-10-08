"""Infrastructure checks: no student exercise implementations required."""
from pathlib import Path
import subprocess
import types

import numpy as np
import pytest

from court_detection import runner
from _helpers import write_video


def court_text():
    # Corners and net poles fit within the 64x64 synthetic video.
    return ('10;10\n10;50\n50;50\n50;10\n10;25\n50;25\n'
            'Homography Matrix:\n[1, 0, 0;\n 0, 1, 0;\n 0, 0, 1]\n')


def assets(tmp_path):
    video = write_video(tmp_path / 'clip_3.mp4', [np.zeros((64, 64, 3), np.uint8)] * 3)
    exe = tmp_path / 'detector'
    exe.touch()
    return video, exe, tmp_path / 'result.txt', tmp_path / 'overlay.png'


def test_existing_court_file_needs_no_detector(tmp_path, monkeypatch):
    source = tmp_path / 'supplied.txt'
    source.write_text(court_text())
    output = tmp_path / 'out/court.txt'
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: pytest.fail('must not run detector'))
    runner.detect_court([], output, tmp_path / 'overlay.png', lambda p: True, court_file=source)
    assert output.read_text() == source.read_text()
    # Repeated use of the output itself is also valid.
    runner.detect_court([], output, tmp_path / 'overlay.png', lambda p: True, court_file=output)


@pytest.mark.parametrize('text', ['', '1;2\n', court_text().replace('1, 0, 0', 'nan, 0, 0'),
                                 court_text().replace('0, 0, 1', '0, 0, 0')])
def test_rejects_malformed_court_file(tmp_path, text):
    path = tmp_path / 'court.txt'
    path.write_text(text)
    with pytest.raises(ValueError, match='Invalid court file'):
        runner.validate_court_file(path)


def test_short_clip_valid_frames_retry_and_no_stale_output(tmp_path, monkeypatch):
    video, exe, output, overlay = assets(tmp_path)
    output.write_text('previous successful output')
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[1] == str(video.resolve())
        assert 0 <= int(command[4]) < 3
        assert kwargs['timeout'] == 2
        if len(calls) == 1:
            Path(command[2]).write_text('partial output')
            return types.SimpleNamespace(returncode=1, stderr='bad fit', stdout='')
        assert not Path(command[2]).exists()
        Path(command[2]).write_text(court_text())
        Path(command[3]).write_bytes(b'overlay')
        return types.SimpleNamespace(returncode=0, stderr='', stdout='')

    monkeypatch.setattr(runner.subprocess, 'run', run)
    runner.detect_court([video], output, overlay, lambda p: True,
                       executable=exe, attempts=3, timeout=2)
    assert len(calls) == 2
    assert output.read_text() == court_text()
    assert overlay.read_bytes() == b'overlay'


@pytest.mark.parametrize('failure', ['timeout', 'outside', 'malformed', 'ordering'])
def test_failures_stop_at_attempt_limit_and_preserve_previous_result(tmp_path, monkeypatch, failure):
    video, exe, output, overlay = assets(tmp_path)
    output.write_text('previous result')
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        text = court_text().replace('50;50', '50;100') if failure == 'outside' else court_text()
        Path(command[2]).write_text('partial' if failure == 'malformed' else text)
        Path(command[3]).write_bytes(b'overlay')
        return types.SimpleNamespace(returncode=0, stderr='', stdout='')

    monkeypatch.setattr(runner.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='failed after 2 attempts'):
        runner.detect_court([video], output, overlay, lambda p: failure != 'ordering',
                           executable=exe, attempts=2, timeout=.1)
    assert len(calls) == 2
    assert output.read_text() == 'previous result'
    assert not list(tmp_path.glob('court-*'))


def test_no_readable_clips_fails_before_subprocess(tmp_path, monkeypatch):
    _, exe, output, overlay = assets(tmp_path)
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: pytest.fail('must not run'))
    with pytest.raises(RuntimeError, match='No readable rally clips'):
        runner.detect_court([], output, overlay, lambda p: True, executable=exe)


def test_missing_detector_has_build_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match='build_court_detector.sh'):
        runner.detect_court([], tmp_path / 'out.txt', tmp_path / 'out.png', lambda p: True,
                           executable=tmp_path / 'missing')


def test_student_todo_is_not_swallowed(tmp_path):
    source = tmp_path / 'court.txt'
    source.write_text(court_text())

    def incomplete(path):
        raise NotImplementedError('TODO [Ex 1.3]')

    with pytest.raises(NotImplementedError, match='1.3'):
        runner.detect_court([], tmp_path / 'out.txt', tmp_path / 'out.png', incomplete,
                           court_file=source)


def test_automatic_detection_propagates_student_todo(tmp_path, monkeypatch):
    video, exe, output, overlay = assets(tmp_path)

    def run(command, **kwargs):
        Path(command[2]).write_text(court_text())
        return types.SimpleNamespace(returncode=0, stderr='', stdout='')

    def incomplete(path):
        raise NotImplementedError('TODO [Ex 1.3]')

    monkeypatch.setattr(runner.subprocess, 'run', run)
    with pytest.raises(NotImplementedError, match='1.3'):
        runner.detect_court([video], output, overlay, incomplete, executable=exe)
