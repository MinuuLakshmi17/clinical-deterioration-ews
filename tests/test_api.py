from fastapi.testclient import TestClient
from app.main import app

def test_health_and_patients():
    with TestClient(app) as client:
        h=client.get('/health')
        assert h.status_code==200
        assert h.json()['status']=='ok'
        p=client.get('/patients')
        assert p.status_code==200
        assert len(p.json()) > 0

def test_risk_endpoint():
    with TestClient(app) as client:
        patients=client.get('/patients').json()
        pid=patients[0]['patient_id']
        r=client.get(f'/patients/{pid}/risk')
        assert r.status_code==200
        body=r.json()
        assert 0 <= body['risk_probability'] <= 1
        assert body['horizon_hours']==6
        assert body['drivers']
