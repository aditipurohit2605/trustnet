import pytest
from pydantic import ValidationError
from trustnet.schemas import Claim, Evidence, ClaimReport, Verdict

def make_report(confidence=0.4):
    claim = Claim(id="c1", text="A dam near the city is about to collapse")
    evidence = [
        Evidence(
            doc_id="d1",
            text="The water resources department reports normal reservoir levels.",
            source="Example State Water Resources Dept",
            source_type="government",
        )
    ]
    return ClaimReport(
        claim=claim,
        evidence=evidence,
        verdict=Verdict.INSUFFICIENT,
        confidence=confidence,
        gaps=["Original date of the viral message could not be established"],
    )

def test_report_builds():
    report = make_report()
    assert report.verdict == Verdict.INSUFFICIENT
    assert len(report.gaps) == 1

def test_confidence_must_be_between_0_and_1():
    with pytest.raises(ValidationError):
        make_report(confidence=1.5)
