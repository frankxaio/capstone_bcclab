#!/usr/bin/env python3
"""Exercise the native detector on a still and a generated video, without model weights."""
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from court_detection.runner import default_executable, validate_court_file


def run(command, expected=0):
    result = subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=120)
    if result.returncode != expected:
        raise RuntimeError(f'Detector returned {result.returncode}: {result.stdout}\n{result.stderr}')


def main():
    detector = default_executable()
    if not detector.is_file():
        raise FileNotFoundError('Run bash scripts/build_court_detector.sh first')
    folder = ROOT / 'court_detection/build/smoke-test'
    folder.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'court_detection/court_image.png'
    frame = cv2.imread(str(source))
    if frame is None:
        raise RuntimeError('Cannot read bundled court image')
    reference, _ = validate_court_file(ROOT / 'court_detection/court.txt')
    video = folder / 'sample.mp4'
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    if not writer.isOpened():
        raise RuntimeError('OpenCV cannot write the smoke-test MP4')
    for _ in range(3):
        writer.write(frame)
    writer.release()
    for name, input_path, extra in [('image', source, []), ('video', video, ['1'])]:
        court = folder / f'{name}.txt'
        overlay = folder / f'{name}.png'
        # Remove only this check's previous artifacts; stale output cannot pass.
        court.unlink(missing_ok=True)
        overlay.unlink(missing_ok=True)
        run([detector, input_path, court, overlay, *extra])
        points, matrix = validate_court_file(court)
        homogeneous = matrix @ np.column_stack([points[:4], np.ones(4)]).T
        projected = (homogeneous[:2] / homogeneous[2]).T
        np.testing.assert_allclose(projected, [[0, 0], [0, 13.41], [6.1, 13.41], [6.1, 0]], atol=1e-3)
        # The sample is annotated, so this is a broad regression check, not a benchmark.
        error = np.linalg.norm(points[:4] - reference[:4], axis=1).max()
        if error > 25:
            raise AssertionError(f'{name}: corners differ from supplied reference by {error:.1f} pixels')
        if cv2.imread(str(overlay)) is None:
            raise AssertionError('Detector did not write a readable overlay')
        print(f'{name}: valid output and camera-to-court homography; max corner difference {error:.1f}px')
    run([detector, folder / 'missing.mp4', folder / 'bad.txt', folder / 'bad.png'], expected=2)
    run([detector, video, folder / 'bad.txt', folder / 'bad.png', '99'], expected=2)
    print(f'Native detector smoke test passed. Inspect overlays in {folder}')


if __name__ == '__main__':
    main()
