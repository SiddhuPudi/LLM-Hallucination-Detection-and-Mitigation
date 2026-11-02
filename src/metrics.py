import re
import string
import unicodedata
from collections import Counter

_TRANS = {ord(c): " " for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212-"}
_TRANS.update({ord(c): None for c in "\u2018\u2019\u201c\u201d"})

NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20",
}
_STOP = {"a", "an", "the", "and", "of"}

def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, gold):
    return float(normalize(pred) == normalize(gold))


def contains_match(pred, gold):
    g = normalize(gold)
    return float(g != "" and g in normalize(pred))


def f1_score(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)

def is_abstain(pred):
    p = normalize(pred)
    phrases = [
        "insufficient evidence", "unknown", "not enough information",
        "i do not know", "i dont know", "cannot be determined", "not mentioned",
    ]
    return any(s in p for s in phrases)

def lenient_tokens(s):
    s = unicodedata.normalize("NFKC", str(s)).lower().translate(_TRANS)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return [NUM_WORDS.get(t, t) for t in s.split() if t not in _STOP]


def lenient_match(pred, gold):
    """Strict 'contains', OR every key word of the gold answer appears in the prediction."""
    if contains_match(pred, gold):
        return 1.0
    g, p = lenient_tokens(gold), lenient_tokens(pred)
    return float(len(g) > 0 and set(g) <= set(p))


def short_form_match(pred, gold):
    """Every word of the prediction appears in the gold answer (a shorter form of it).
    NOT counted as correct automatically; flagged for manual review."""
    g, p = lenient_tokens(gold), lenient_tokens(pred)
    return float(len(p) > 0 and set(p) <= set(g))