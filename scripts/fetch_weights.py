#!/usr/bin/env python3
"""Download pinned weights and verify their hashes, including for ZIP exports without Git."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / 'scripts/weights.json').read_text())


def verified(path, item):
    if not path.is_file() or path.stat().st_size != item['size']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == item['sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='include optional training checkpoints')
    parser.add_argument('--check', action='store_true', help='verify only; do not download')
    args = parser.parse_args()
    failures = []
    for item in MANIFEST['files']:
        if not args.all and not item['required']:
            continue
        path = ROOT / item['path']
        if verified(path, item):
            print(f"OK {item['path']}")
            continue
        if args.check:
            failures.append(item['path'])
            continue
        url = (f"https://media.githubusercontent.com/media/{MANIFEST['repository']}/"
               f"{MANIFEST['revision']}/{item['path']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.download')
        print(f"Downloading {item['path']} ({item['size'] / 1e6:.1f} MB)", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as out:
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    out.write(chunk)
            if not verified(temporary, item):
                raise ValueError('Downloaded file does not match the pinned size and SHA-256')
            temporary.replace(path)
        except Exception as error:
            print(f"ERROR {item['path']}: {error}", file=sys.stderr)
            failures.append(item['path'])
        finally:
            temporary.unlink(missing_ok=True)
    if failures:
        print('Missing or invalid weights: ' + ', '.join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
