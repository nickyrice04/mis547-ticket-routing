"""The shared cleaning step, used identically at training and at inference time."""
from common import clean


def test_clean_joins_lowercases_and_strips_links_and_emails():
    t = clean("Refund REQUEST", "See https://example.com or mail me@bank.com about TICKET-12345.")
    assert t == "refund request . see or mail about ticket-12345."
    assert "http" not in t and "@" not in t


def test_ticket_ids_survive_cleaning_as_trained():
    # the id pattern runs after lowercasing and never matches, see the comment in common.clean
    assert "ticket-12345" in clean("", "about TICKET-12345")


def test_clean_handles_missing_parts():
    assert clean(None, "body only") == ". body only"
    assert clean("", "") == "."
