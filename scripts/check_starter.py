#!/usr/bin/env python3
"""Instructor check: only intentional TODO failures are allowed in a starter release."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix='badminton-starter-') as folder:
        report = Path(folder) / 'tests.xml'
        result = subprocess.run([sys.executable, '-m', 'pytest', '--tb=short', f'--junitxml={report}'],
                                cwd=ROOT, capture_output=True, text=True)
        if not report.exists():
            print(result.stdout + result.stderr)
            return 1
        cases = ET.parse(report).findall('.//testcase')
        todo, passed, problems = 0, 0, []
        for case in cases:
            failure = case.find('failure')
            if case.find('error') is not None or case.find('skipped') is not None:
                problems.append(case.attrib.get('name', 'collection error'))
            elif failure is not None:
                if re.fullmatch(r'NotImplementedError: TODO \[Ex \d+\.\d+[a-z]?\]', failure.attrib.get('message', '')):
                    todo += 1
                else:
                    problems.append(case.attrib['name'])
            else:
                passed += 1
        # Collection errors may be reported at suite level instead of as test cases.
        errors = sum(int(s.attrib.get('errors', 0)) for s in ET.parse(report).findall('.//testsuite'))
        if problems or errors or not cases or result.returncode not in (0, 1):
            print(result.stdout + result.stderr)
            print('Unexpected starter failures/skips:', ', '.join(problems))
            return 1
        print(f'Starter verified: {passed} passed, {todo} intentional TODO failures; no errors or skips.')
        print('This checks starter infrastructure, not completed solutions or full-video accuracy.')
        return 0


if __name__ == '__main__':
    sys.exit(main())
