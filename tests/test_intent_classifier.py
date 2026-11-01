from leadforge_intent_classifier.experiment import _load_leads, _demo_predict, train_model, evaluate


def test_load_leads():
    leads = _load_leads()
    assert len(leads) >= 1


def test_demo_mode():
    leads = _load_leads()
    label = _demo_predict(leads[0].text)
    assert label in {"hot", "warm", "cold"}


def test_trained_model():
    leads = _load_leads()
    vectorizer, clf = train_model(leads)
    records, summary = evaluate(vectorizer, clf, leads)
    assert summary["accuracy"] >= 0.0
    assert summary["f1"] >= 0.0
    assert len(records) == len(leads)
