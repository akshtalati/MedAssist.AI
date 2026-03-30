from src.response_contract import normalize_contract, validate_contract


def test_normalize_contract_fills_required_fields():
    payload = normalize_contract({"summary": "ok"})
    ok, missing = validate_contract(payload)
    assert ok is True
    assert missing == []
    assert "confidence" in payload
    assert "insufficient_evidence" in payload

