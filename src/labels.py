from metrics import lenient_match, short_form_match, is_abstain, lenient_tokens

# Short-form answers judged INCOMPLETE by hand (not counted as correct). Any short-form answer to a
# question starting with one of these prefixes is incomplete (dev, validation and test verdicts).
INCOMPLETE = (
    "Dharmatama",
    "When was the movie released in 2007",
    "What job do both Herbert Selpin",
    'When was the writer of "In This Life" born',
    "What is the name of the world's largest people group",
)

# Answer-specific verdicts: (question prefix, incomplete answer). Other answers to the same
# question are judged normally.
INCOMPLETE_ANSWERS = (
    ("When was the inventor of the wobble board born", "1930"),
)


def correct_lenient(pred, gold):
    return bool(lenient_match(pred, gold)) and not is_abstain(pred)


def correct_verified(pred, gold, question):
    """Lenient match, or a short-form answer that was not judged incomplete."""
    if is_abstain(pred):
        return False
    if lenient_match(pred, gold):
        return True
    if not short_form_match(pred, gold):
        return False
    q = str(question)
    if q.startswith(INCOMPLETE):
        return False
    p = " ".join(lenient_tokens(pred))
    for prefix, ans in INCOMPLETE_ANSWERS:
        if q.startswith(prefix) and p == " ".join(lenient_tokens(ans)):
            return False
    return True