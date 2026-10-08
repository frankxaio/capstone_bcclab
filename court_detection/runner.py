"""Bounded, headless court detection for the full pipeline."""
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def default_executable():
    return ROOT / 'court_detection/build/court-detection'


def validate_court_file(path):
    """Check the external file format, finite coordinates and invertible homography.

    The student's Ex 1.3 checks the corner ordering separately.
    """
    lines = Path(path).read_text().splitlines()
    try:
        points = np.array([[float(v) for v in line.split(';')] for line in lines[:6]])
        start = lines.index('Homography Matrix:') + 1
        rows = [line.replace('[', '').replace(']', '').replace(';', '').split(',')
                for line in lines[start:start + 3]]
        matrix = np.array(rows, dtype=float)
        if points.shape != (6, 2) or matrix.shape != (3, 3):
            raise ValueError('expected six points and a 3x3 homography')
        if not np.isfinite(points).all() or not np.isfinite(matrix).all():
            raise ValueError('coordinates and homography must be finite')
        if np.linalg.matrix_rank(matrix) != 3:
            raise ValueError('homography must be invertible')
    except (ValueError, IndexError) as error:
        raise ValueError(f'Invalid court file {path}: {error}') from error
    return points, matrix


def detect_court(clips, output, image, check_corners, *, executable=None,
                 court_file=None, attempts=5, timeout=120):
    """Use supplied coordinates or try valid frames from existing rally clips."""
    output, image = Path(output), Path(image)
    if attempts < 1 or timeout <= 0:
        raise ValueError('Court attempts and timeout must be positive')
    output.parent.mkdir(parents=True, exist_ok=True)
    image.parent.mkdir(parents=True, exist_ok=True)
    if court_file is not None:
        source = Path(court_file)
        validate_court_file(source)
        if not check_corners(source):
            raise ValueError(f'Implausible court corners in {source}')
        if source.resolve() != output.resolve():
            shutil.copyfile(source, output)
        return output

    detector = Path(executable) if executable is not None else default_executable()
    if not detector.is_file():
        raise FileNotFoundError(f'Court detector missing: {detector}. '
                                'Run bash scripts/build_court_detector.sh or use --court-file.')
    candidates = []
    for clip in sorted(map(Path, clips)):
        cap = cv2.VideoCapture(str(clip))
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        if count > 0 and width > 0 and height > 0:
            candidates.append((clip, count, width, height))
    if not candidates:
        raise RuntimeError('No readable rally clips for court detection')

    errors = []
    # Round-robin over clips, then over evenly spaced valid frames in each clip.
    for attempt in range(attempts):
        clip, count, width, height = candidates[attempt % len(candidates)]
        round_index = attempt // len(candidates)
        rounds = (attempts + len(candidates) - 1) // len(candidates)
        frame = min(count - 1, int(count * (round_index + 1) / (rounds + 1)))
        with tempfile.TemporaryDirectory(prefix='court-', dir=output.parent) as folder:
            temporary = Path(folder) / 'court.txt'
            overlay = Path(folder) / 'court.png'
            command = [str(detector.resolve()), str(clip.resolve()), str(temporary.resolve()),
                       str(overlay.resolve()), str(frame)]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
                if result.returncode:
                    raise RuntimeError(f'exit {result.returncode}: {(result.stderr or result.stdout)[-1500:]}')
                points, _ = validate_court_file(temporary)
                corners = points[:4]
                if (corners < 0).any() or (corners[:, 0] >= width).any() or (corners[:, 1] >= height).any():
                    raise ValueError('Court corners fall outside the video frame')
                if not check_corners(temporary):
                    raise ValueError('Implausible court corners')
                if not overlay.is_file():
                    raise ValueError('Detector did not write an overlay image')
                shutil.copyfile(overlay, image)
                temporary.replace(output)
                return output
            except NotImplementedError:
                raise  # Student TODOs are not detector failures.
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                errors.append(f'{clip.name}, frame {frame}: {error}')
    raise RuntimeError(f'Court detection failed after {attempts} attempts. '
                       'Inspect the video or provide --court-file.\n' + '\n'.join(errors))
