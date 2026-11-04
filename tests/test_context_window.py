from types import SimpleNamespace

from pipeline.context import retain_latest_user_turn


def test_retain_latest_user_turn_drops_prior_conversation():
    system = SimpleNamespace(role="system")
    old_user = SimpleNamespace(role="user")
    old_assistant = SimpleNamespace(role="assistant")
    latest_user = SimpleNamespace(role="user")

    retained = retain_latest_user_turn([system, old_user, old_assistant, latest_user], latest_user)

    assert retained == [system, latest_user]
