#!/usr/bin/env python3
"""Run a completed exercise pipeline from the repository root."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--court-file', type=Path, help='verified coordinates for this video/camera')
    parser.add_argument('--court-executable', type=Path)
    parser.add_argument('--court-attempts', type=int, default=5)
    parser.add_argument('--court-timeout', type=float, default=120, help='seconds per attempt')
    args = parser.parse_args()
    video = args.video.resolve()
    if not video.is_file():
        parser.error(f'Video not found: {video}')
    if args.court_attempts < 1 or args.court_timeout <= 0:
        parser.error('Court attempts and timeout must be positive')
    court_file = args.court_file.resolve() if args.court_file else None
    executable = args.court_executable.resolve() if args.court_executable else None
    from fetch_weights import MANIFEST, verified
    missing = [item['path'] for item in MANIFEST['files']
               if item['required'] and not verified(ROOT / item['path'], item)]
    if missing:
        parser.error('Run python scripts/fetch_weights.py first. Invalid weights: ' + ', '.join(missing))
    from court_detection.runner import default_executable, validate_court_file
    if court_file:
        try:
            validate_court_file(court_file)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    elif not (executable or default_executable()).is_file():
        parser.error('Build the court detector or provide --court-file. See court_detection/README.md.')
    os.chdir(ROOT)
    from main import main as run
    run(video, court_executable=executable, court_file=court_file,
        court_attempts=args.court_attempts, court_timeout=args.court_timeout)


if __name__ == '__main__':
    main()
