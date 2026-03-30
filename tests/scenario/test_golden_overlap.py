import json
from pathlib import Path


def _topk_overlap(expected: list[str], actual: list[str], k: int = 5) -> float:
    e = set(expected[:k])
    a = set(actual[:k])
    if not e:
        return 1.0
    return len(e.intersection(a)) / len(e)


def test_golden_topk_overlap_threshold():
    fixture_path = Path(__file__).resolve().parent / "golden_cases.json"
    payload = json.loads(fixture_path.read_text())
    for case in payload["cases"]:
        score = _topk_overlap(case["expected_top"], case["actual_top"], k=case.get("k", 5))
        assert score >= case.get("min_overlap", 0.5), f"Low overlap for case {case['id']}: {score}"

