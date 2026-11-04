from types import SimpleNamespace

import pytest

from pipeline.jev_location import JevLocationClient, build_jev_location_state


def test_parse_location_status_provided():
    result = JevLocationClient._parse_response({
        "model": "typesafe/jev-1.14",
        "answers": {
            "location_status": {
                "type": "choice",
                "choice": "provided",
                "confidence": 0.94,
            }
        },
    })
    assert result.status == "provided"
    assert result.confidence == 0.94


@pytest.mark.parametrize("status", ["not_provided", "unclear"])
def test_parse_non_provided_location_statuses(status):
    result = JevLocationClient._parse_response({
        "answers": {
            "location_status": {
                "type": "choice",
                "choice": status,
                "confidence": 0.9,
            }
        }
    })
    assert result.status == status


def test_parse_rejects_unknown_status():
    with pytest.raises(ValueError):
        JevLocationClient._parse_response({
            "answers": {
                "location_status": {
                    "type": "choice",
                    "choice": "maybe",
                    "confidence": 0.9,
                }
            }
        })


def test_build_state_includes_pending_location_and_latest_transcript():
    session = SimpleNamespace(
        pending_field="location",
        next_missing_field="location",
        stage="location",
        recent_turns=[],
        to_context_dict=lambda: {"facts": {"location": None}},
    )
    state = build_jev_location_state(session, "నాకు విజయవాడలో కావాలి")
    assert state["pending_qualification_field"] == "location"
    assert state["latest_caller_utterance"] == "నాకు విజయవాడలో కావాలి"
