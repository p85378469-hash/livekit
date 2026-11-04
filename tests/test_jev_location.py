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


def test_build_state_contains_only_latest_caller_utterance():
    latest = "caller answer"
    assert build_jev_location_state(latest) == {"latest_caller_utterance": latest}
