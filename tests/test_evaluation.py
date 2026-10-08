"""Event-count regressions that need only the CPU exercise dependencies."""
import pandas as pd
import pytest

from scripts.evaluate import evaluate_hit_detection, match_hit_events, detect_player_mapping


def predictions(intervals):
    return pd.DataFrame([(a, b, 'Top_smash', '0') for a, b in intervals],
                        columns=['start_frame', 'end_frame', 'stroke', 'player'])


def truth(frames):
    return pd.DataFrame({'frame_num': frames, 'type': ['smash'] * len(frames),
                         'player': ['A'] * len(frames)})


@pytest.mark.parametrize('intervals,frames,counts', [
    ([(10, 20), (30, 30)], [10, 20, 30], (1, 1, 2)),
    ([(10, 10), (10, 10)], [10], (1, 1, 0)),
    ([(9, 11), (19, 21)], [10, 20], (2, 0, 0)),
    ([(9, 11)], [10, 10], (0, 1, 2)),
    ([], [10, 20], (0, 0, 2)),
    ([(10, 10)], [], (0, 1, 0)),
    ([], [], (0, 0, 0)),
    ([(0, 0)], [0], (1, 0, 0)),
])
def test_one_to_one_counts(intervals, frames, counts):
    result = evaluate_hit_detection(predictions(intervals), truth(frames))
    tp, fp, fn = counts
    assert (result['TP'], result['FP'], result['FN']) == counts
    assert result['recall'] == pytest.approx(tp / len(frames) if frames else 0)
    assert result['precision'] == pytest.approx(tp / len(intervals) if intervals else 0)
    assert result['f1'] == pytest.approx(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0)
    assert 'accuracy' not in result


def test_matching_independent_of_order_and_dataframe_index():
    pred = predictions([(10, 20), (9, 11), (20, 20)])
    pred.index = [8, 8, 2]
    gt = truth([20, 10])
    assert match_hit_events(pred, gt) == [(1, 1), (2, 0)]
    assert evaluate_hit_detection(pred, gt) == evaluate_hit_detection(pred.iloc[::-1], gt.iloc[::-1])


@pytest.mark.parametrize('intervals,frames', [([(20, 10)], [10]),
    ([(float('nan'), 20)], [10]), ([(0, 20)], [float('inf')]),
    ([(-1, 20)], [10]), ([(0, 20)], [10.5])])
def test_invalid_frames_are_rejected(intervals, frames):
    with pytest.raises(ValueError):
        match_hit_events(predictions(intervals), truth(frames))


def test_player_mapping_uses_only_eligible_matches():
    pred = predictions([(10, 20), (30, 30)])
    pred.loc[1, 'player'] = 1
    assert detect_player_mapping(pred, truth([10, 20, 30])) == {'0': 'B', '1': 'A', 'X': 'X'}
