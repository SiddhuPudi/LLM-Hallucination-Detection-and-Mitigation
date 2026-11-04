from metrics import lenient_match, short_form_match, is_abstain

# Short-form answers judged INCOMPLETE by hand (not counted as correct).
# Edit this list if you disagree with a verdict in Appendix A or A2 of the report.
INCOMPLETE = ("Dharmatama", "When was the movie released in 2007", "What job do both Herbert Selpin")


def correct_lenient(pred, gold):
    return bool(lenient_match(pred, gold)) and not is_abstain(pred)


def correct_verified(pred, gold, question):
    """Lenient match, or a short-form answer that was judged correct."""
    if is_abstain(pred):
        return False
    if lenient_match(pred, gold):
        return True
    return bool(short_form_match(pred, gold)) and not str(question).startswith(INCOMPLETE)