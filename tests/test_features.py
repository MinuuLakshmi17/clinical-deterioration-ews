import pandas as pd
from app.ml.features import engineer_features, prepare_matrix

def test_temporal_features_exist():
    df = pd.DataFrame({"patient_id":["P1"]*4,"timestamp":pd.date_range("2026-01-01",periods=4,freq="h"),
        "age":[50]*4,"sex":["F"]*4,"comorbidity_burden":[1]*4,
        "heart_rate":[70,75,80,90],"sbp":[120]*4,"dbp":[70]*4,"resp_rate":[16]*4,
        "spo2":[98]*4,"temperature":[36.8]*4,"lactate":[1.1]*4,"wbc":[8]*4,"creatinine":[1]*4,"oxygen_flow":[1]*4})
    out=engineer_features(df)
    assert out.loc[3,"heart_rate_delta3h"] == 20
    assert "shock_index" in out
    assert prepare_matrix(df).shape == (4, 46)
