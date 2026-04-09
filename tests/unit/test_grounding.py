from src.grounding import validate_grounding


def test_grounding_removes_untrusted_links():
    answer = "- [Good](https://a.com)\n- [Bad](https://evil.com)"
    cleaned, violations = validate_grounding(answer, {"https://a.com"})
    assert violations == 1
    assert "https://evil.com" not in cleaned
    assert "Good" in cleaned

