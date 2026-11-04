from ner.data import convert_tags
from ner.generate_conversational import make_row


def test_location_only_tag_mapping():
    assert convert_tags([0, 1, 2, 3, 4, 5, 6]) == [0, 0, 0, 0, 0, 1, 2]


def test_multitoken_location_bio_labels():
    row = make_row("Nenu New Vijayawada lo illu konali", "New Vijayawada")
    assert row["tokens"] == ["Nenu", "New", "Vijayawada", "lo", "illu", "konali"]
    assert row["location_tags"] == [0, 1, 2, 0, 0, 0]


def test_tanglish_single_token_location():
    row = make_row("Nenu Vijayawada lo illu konali", "Vijayawada")
    assert row["location_tags"] == [0, 1, 0, 0, 0]
