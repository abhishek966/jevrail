from jevrail.scanners import find_sensitive, redact


def test_redacts_email_ssn_card_and_api_key():
    text = (
        "email user@example.com ssn 123-45-6789 "
        "card 4111111111111111 key sk-abcdefghijklmnopqrstuvwxyz"
    )
    spans = find_sensitive(text)
    labels = {span.label for span in spans}
    assert {"email", "ssn", "credit_card", "api_key"} <= labels
    masked = redact(text, spans)
    assert "user@example.com" not in masked
    assert "123-45-6789" not in masked
    assert "4111111111111111" not in masked
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in masked


def test_ignores_a_number_that_fails_luhn():
    spans = find_sensitive("not a card 1234567890123456")
    assert all(span.label != "credit_card" for span in spans)
