#!/usr/bin/env python3
"""Check the student environment without running unfinished exercises."""
import argparse
import ast
import importlib
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline', action='store_true', help='also check full-pipeline dependencies and weights')
    parser.add_argument('--court-file', type=Path, help='use existing court coordinates instead of a detector')
    args = parser.parse_args()
    problems = []
    if sys.version_info[:2] != (3, 11):
        problems.append('Use Python 3.11 for the tested exercise environment')
    packages = ['pytest', 'numpy', 'pandas', 'torch', 'cv2', 'einops', 'skimage', 'sklearn', 'yaml', 'tqdm']
    if args.pipeline:
        packages += ['torchvision', 'mmpose.apis', 'mmcv.ops', 'mmdet', 'moviepy.editor',
                     'ultralytics', 'supervision', 'sports', 'matplotlib', 'seaborn', 'parse', 'torchinfo']
    for name in packages:
        try:
            importlib.import_module(name)
            print(f'OK import {name}')
        except Exception as error:
            problems.append(f'{name}: {error}')
    todo_ids, test_ids = set(), set()
    for path in ROOT.rglob('*.py'):
        if any(part.startswith('.venv') or part in {'venv', 'build', '.git'}
               for part in path.relative_to(ROOT).parts):
            continue
        source = path.read_text(encoding='utf-8-sig')
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as error:
            problems.append(str(error))
            continue
        if 'tests' in path.relative_to(ROOT).parts:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and ast.unparse(node.func) == 'pytest.mark.ex':
                    test_ids.update(arg.value for arg in node.args if isinstance(arg, ast.Constant))
        else:
            todo_ids.update(re.findall(r'TODO \[Ex ([0-9]+\.[0-9]+[a-z]?)\]', source))
    if todo_ids - test_ids:
        problems.append(f'Exercises without tests: {sorted(todo_ids - test_ids)}')
    for name in ['hitnet', 'tempose', 'tracknet']:
        try:
            import numpy as np
            with np.load(ROOT / f'tests/fixtures/{name}.npz') as fixture:
                if not fixture.files:
                    problems.append(f'Empty fixture: {name}')
        except Exception as error:
            problems.append(f'Fixture {name}: {error}')
    print(f'{len(todo_ids)} exercise IDs in source, {len(test_ids)} tested IDs')
    if args.pipeline:
        from fetch_weights import MANIFEST, verified
        for item in MANIFEST['files']:
            if item['required'] and not verified(ROOT / item['path'], item):
                problems.append(f"Missing/invalid {item['path']}; run python scripts/fetch_weights.py")
        try:
            from court_detection.runner import default_executable, validate_court_file
            if args.court_file:
                validate_court_file(args.court_file)
            elif not default_executable().is_file():
                problems.append('Court detector missing; run bash scripts/build_court_detector.sh')
            import torch
            if not torch.cuda.is_available():
                problems.append('Full pipeline currently requires CUDA for MMPose')
            else:
                # Driver visibility alone does not prove this wheel supports the GPU.
                probe = torch.ones(8, device='cuda').square().sum().item()
                if probe != 8:
                    problems.append('CUDA computation returned an unexpected result')
                else:
                    print(f'OK CUDA computation on {torch.cuda.get_device_name(0)}')
        except Exception as error:
            problems.append(f'Pipeline: {error}')
    for problem in problems:
        print('ERROR ' + problem, file=sys.stderr)
    if problems:
        return 1
    print('Setup checks passed. Exercise TODOs are intentional; run python -m pytest next.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
