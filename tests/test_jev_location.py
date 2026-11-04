from pipeline.jev_location import JevLocationClient, build_jev_location_state


def test_build_state_contains_only_latest_caller_utterance():
    latest = "caller answer"
    assert build_jev_location_state(latest) == {"latest_caller_utterance": latest}
