from fastapi.testclient import TestClient
from digit_classifier.api import app

client=TestClient(app)


def test_health():
    r=client.get('/health')
    assert r.status_code==200
    assert r.json()['status'] in ['ok','model_not_trained']


def test_reject_invalid_input():
    assert client.post('/predict',json={'pixels':[0]*63}).status_code==422
    assert client.post('/predict',json={'pixels':[0]*63+[99]}).status_code==422


def test_examples():
    d=client.get('/examples').json()
    assert len(d)==10
    assert all(len(x['pixels'])==64 for x in d)


def test_predict_after_training():
    r=client.get('/examples')
    ex=r.json()[0]
    out=client.post('/predict',json={'pixels':ex['pixels']})
    assert out.status_code==200
    payload=out.json()
    assert isinstance(payload['digit'],int)
    assert 0<=payload['confidence']<=1
    assert len(payload['probabilities'])==10
