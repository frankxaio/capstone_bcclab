#!/usr/bin/env python3
"""Build a student ZIP without videos, weights, build outputs, or instructor answers."""
import argparse
import hashlib
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {'HitNet', 'MMPose', 'TemPose', 'TrackNetV3', 'ai_badminton',
           'court_detection', 'data', 'docs', 'rally_clipping', 'scripts',
           'team_classifier', 'tests'}
ROOT_FILES = {'README.md', 'LICENSE', 'main.py', 'utils.py', 'pytest.ini',
              'requirements.txt', 'requirements-exercises.txt', '.gitignore', '.gitattributes'}
EXCLUDED_DIRS = {'__pycache__', 'build', 'videos', 'results', 'final_result',
                 'npy', 'traj', 'clips', 'exp', 'weight', '.pytest_cache'}
EXTENSIONS = {'.py', '.md', '.txt', '.csv', '.npz', '.png', '.jpg', '.jpeg',
              '.json', '.yml', '.yaml', '.sh', '.cpp', '.h'}


def student_files(root=ROOT):
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if not path.is_file() or path.is_symlink():
            continue
        if len(relative.parts) == 1:
            if path.name in ROOT_FILES:
                yield path
        elif (relative.parts[0] in FOLDERS
              and not any(part in EXCLUDED_DIRS or part.startswith('.') for part in relative.parts)
              and (path.suffix in EXTENSIONS or path.name in {'LICENSE', 'LICENCE'})):
            yield path


def build_archive(output, root=ROOT):
    paths = list(student_files(root))
    source = '\n'.join(path.read_text(encoding='utf-8-sig') for path in paths
                       if path.suffix == '.py' and path.relative_to(root).parts[0] != 'tests')
    todos = set(re.findall(r'raise NotImplementedError\("TODO \[Ex ([\d.]+[a-z]?)\]"\)', source))
    if len(todos) != 43:
        raise ValueError(f'Refusing student release: expected 43 unfinished exercise IDs, found {len(todos)}')
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                info = zipfile.ZipInfo('BadmintonAI-End2End-exercises/' + path.relative_to(root).as_posix())
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o100644 << 16)
                archive.writestr(info, path.read_bytes())
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError('Archive integrity check failed')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + '.sha256').write_text(f'{digest}  {output.name}\n')
    print(f'{output}: {len(paths)} files, {output.stat().st_size / 1024**2:.2f} MiB, 43 exercise TODOs')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT.parent / 'releases' / 'BadmintonAI-student.zip')
    build_archive(parser.parse_args().output)
