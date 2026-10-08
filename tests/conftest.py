"""pytest configuration for the exercise tests.

    pytest                     run every test
    pytest --ex 1.4            only the tests of Ex 1.4a/b/c
    pytest --ex 2 --ex 3.1a    all of Process II + Ex 3.1a

At the end a table shows, for every exercise, whether it passes, fails, is still a TODO or is
blocked by another exercise that must be implemented first.
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _helpers import install_stubs  # noqa: E402

install_stubs()

_TODO = re.compile(r'TODO \[Ex ([0-9]+\.[0-9]+[a-z]?)\]')
_RESULTS = defaultdict(list)   # exercise id -> [(nodeid, status, detail)]


def pytest_addoption(parser):
    parser.addoption('--ex', action='append', default=[], metavar='ID',
                     help='only run the tests of these exercises, e.g. --ex 1.4 --ex 3.5a --ex 2')


def pytest_configure(config):
    config.addinivalue_line('markers', 'ex(*ids): exercise id(s) checked by the test, e.g. ex("1.4c")')


def _selected(ex_id, wanted):
    for w in wanted:
        if '.' not in w and ex_id.split('.')[0] == w:                      # a whole part: --ex 2
            return True
        if ex_id == w or (ex_id.startswith(w) and ex_id[len(w):].isalpha()):  # --ex 1.4 -> 1.4a, 1.4b
            return True
    return False


def _ex_ids(item):
    return [i for m in item.iter_markers('ex') for i in m.args]


def pytest_collection_modifyitems(config, items):
    wanted = config.getoption('--ex')
    if not wanted:
        return
    keep, drop = [], []
    for item in items:
        (keep if any(_selected(i, wanted) for i in _ex_ids(item)) else drop).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    ids = _ex_ids(item)
    if not ids:
        return
    if rep.when == 'call' or (rep.when == 'setup' and not rep.passed):
        if rep.passed:
            status, detail = 'PASS', ''
        elif rep.skipped:
            reason = rep.longrepr[2] if isinstance(rep.longrepr, tuple) else str(rep.longrepr)
            status, detail = 'SKIP', reason.replace('Skipped: ', '')
        else:
            status, detail = 'FAIL', ''
            exc = call.excinfo
            if exc is not None and exc.errisinstance(NotImplementedError):
                m = _TODO.search(str(exc.value))
                if m:
                    todo = m.group(1)
                    status, detail = ('TODO', 'not implemented yet') if todo in ids else \
                        ('BLOCKED', f'implement Ex {todo} first')
        for i in ids:
            _RESULTS[i].append((item.nodeid, status, detail))


def _key(ex_id):
    part, rest = ex_id.split('.')
    num = re.match(r'\d+', rest).group()
    return int(part), int(num), rest[len(num):]


def pytest_terminal_summary(terminalreporter):
    if not _RESULTS:
        return
    tr = terminalreporter
    tr.section('Exercise results')
    order = ['FAIL', 'TODO', 'BLOCKED', 'SKIP', 'PASS']
    totals = defaultdict(int)
    for ex_id in sorted(_RESULTS, key=_key):
        rows = _RESULTS[ex_id]
        statuses = [s for _, s, _ in rows]
        status = next(s for s in order if s in statuses)
        passed = statuses.count('PASS')
        details = sorted({d for _, s, d in rows if s == status and d})
        totals[status] += 1
        colour = {'PASS': {'green': True}, 'FAIL': {'red': True}, 'TODO': {'yellow': True},
                  'BLOCKED': {'yellow': True}}.get(status, {})
        tr.write(f'  Ex {ex_id:<6} ')
        tr.write(f'{status:<8}', bold=True, **colour)
        tr.write_line(f' {passed}/{len(rows)} tests passed' + (f'  - {"; ".join(details)}' if details else ''))
    tr.write_line('')
    tr.write_line('  ' + ', '.join(f'{totals[s]} {s.lower()}' for s in order if totals[s]) + ' (exercises)')
