from app.ml.synthetic import generate_synthetic_ehr

def test_generator_is_reproducible():
    a=generate_synthetic_ehr(4,10,seed=7)
    b=generate_synthetic_ehr(4,10,seed=7)
    assert a.equals(b)
    assert len(a)==40
    assert set(a.deterioration_6h.unique()).issubset({0,1})
