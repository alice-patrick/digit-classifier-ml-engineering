import numpy as np
import pytest
from digit_classifier.modeling import load_splits,build_candidates,validate_pixels
from digit_classifier.train import metric_set


def test_splits_disjoint_and_stratified():
    d=load_splits()
    assert len(d.y_train)+len(d.y_validation)+len(d.y_test)==1797
    assert not (set(d.ids_train) & set(d.ids_validation))
    assert not (set(d.ids_train) & set(d.ids_test))
    assert not (set(d.ids_validation) & set(d.ids_test))
    assert len(set(d.y_train))==len(set(d.y_validation))==len(set(d.y_test))==10


def test_pixel_constraints():
    validate_pixels([0.0]*64)
    validate_pixels([16.0]*64)
    for v in ([0.]*63,[0.]*63+[17.],[0.]*63+[float('nan')]):
        with pytest.raises(ValueError):validate_pixels(v)


def test_models_can_predict_probabilities():
    d=load_splits()
    for name,model in build_candidates().items():
        model.fit(d.x_train[:180],d.y_train[:180])
        out=model.predict_proba(d.x_validation[:3])
        assert out.shape==(3,10),name
        np.testing.assert_allclose(out.sum(axis=1),np.ones(3),atol=1e-6)


def test_metric_set_has_valid_bounds():
    p=np.eye(10)[np.arange(10)]*0.9+0.01
    y=np.arange(10)
    m=metric_set(y,y,p)
    assert m['macro_f1']==1.0
    assert m['accuracy']==1.0
