"""Regression checks for test infrastructure, independent of exercise solutions."""
import pytest

from _helpers import _Missing, import_repo


def test_missing_dependency_can_be_subclassed_but_not_instantiated():
    class Transform(_Missing('torchvision.transforms.v2.Transform')):
        pass

    with pytest.raises(pytest.skip.Exception, match='torchvision'):
        Transform()


def test_tempose_dataset_imports_without_optional_transforms():
    module = import_repo('TemPose.data.dataset', 'torch')
    assert len(module.get_stroke_types()) == 35
