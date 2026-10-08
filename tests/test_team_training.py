"""Infrastructure checks: team-classifier training is bounded and uses this video's court."""
import pytest

from _helpers import import_repo


class FakeYOLO:
    def __init__(self, path):
        pass

    def to(self, device):
        return self


class FakeClassifier:
    def __init__(self, **kwargs):
        self.crops = None

    def fit(self, crops):
        self.crops = crops


def train(tmp_path, monkeypatch, crops_per_round):
    mod = import_repo('team_classifier.sport_player_team_classifier')
    rounds = []

    def fake_crops(model, rally_dir, frames_per_rally):
        rounds.append(rally_dir)
        return [object()] * crops_per_round

    monkeypatch.setattr(mod, 'YOLO', FakeYOLO)
    monkeypatch.setattr(mod, 'TeamClassifier', FakeClassifier)
    monkeypatch.setattr(mod, 'get_crops_from_rallies', fake_crops)
    for name in ['ROI_X1', 'ROI_Y1', 'ROI_X2', 'ROI_Y2']:
        monkeypatch.setattr(mod, name, getattr(mod, name))   # restore the ROI after the test
    court = tmp_path / 'court.txt'
    court.write_text('10;20\n5;80\n95;80\n90;20\n')
    return mod, rounds, lambda: mod.train_yolo(tmp_path, court_file=court)


@pytest.mark.parametrize('per_round, expected_rounds, expected_crops', [
    (1500, 3, 4500),   # stops sampling once the target is reached
    (200, 5, 1000),    # too few for the target, but enough to fit after the last round
])
def test_training_samples_a_bounded_number_of_rounds(tmp_path, monkeypatch, per_round,
                                                      expected_rounds, expected_crops):
    mod, rounds, run = train(tmp_path, monkeypatch, per_round)
    classifier = run()
    assert len(rounds) == expected_rounds
    assert len(classifier.crops) == expected_crops
    # The ROI comes from this video's top-left and bottom-right corners.
    assert (mod.ROI_X1, mod.ROI_Y1, mod.ROI_X2, mod.ROI_Y2) == (10, 20, 95, 80)


def test_training_stops_when_no_players_are_found(tmp_path, monkeypatch):
    mod, rounds, run = train(tmp_path, monkeypatch, 10)
    with pytest.raises(RuntimeError, match='player crops'):
        run()
    assert len(rounds) == mod.CROP_ROUNDS
