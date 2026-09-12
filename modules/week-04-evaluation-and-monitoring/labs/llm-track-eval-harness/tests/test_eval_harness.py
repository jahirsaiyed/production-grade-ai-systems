from eval_harness import run


def test_run_produces_report_with_citation_match_rate():
    report = run()

    assert "metrics" in report
    assert "citation_match_rate" in report["metrics"]
    assert 0.0 <= report["metrics"]["citation_match_rate"] <= 1.0
    assert len(report["results"]) == report["n_examples"]
    assert report["n_examples"] == 8
