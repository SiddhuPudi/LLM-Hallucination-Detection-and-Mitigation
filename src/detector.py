import re
import numpy as np
import pandas as pd
from metrics import lenient_tokens

STOP = set("""is are was were be been being has have had do does did to in on at by for with as
that which who whom whose what when where it its this these those his her their he she they
them from into than then both also not no yes more most""".split())


def lex_support(claim, question, evidence):
    q = set(lenient_tokens(question))
    novel = [t for t in lenient_tokens(claim) if t not in q and t not in STOP]
    if not novel:
        return np.nan                       # claim adds no new words: rely on NLI
    ev = set(lenient_tokens(evidence))
    return sum(t in ev for t in novel) / len(novel)


def load_scored(split):
    """Verified claims plus the support scores (retrieved evidence only: this is what the
    deployed detector can see). Hallucination score = 1 - support."""
    v = pd.read_json(f"data/verified_{split}.jsonl", lines=True)
    b = pd.read_json(f"data/rag_b_k3_{split}.jsonl", lines=True).set_index("id")
    ev = {int(i): re.sub(r"\[\d+\]\s*", "", e) for i, e in b["evidence"].items()}
    v["lex"] = [lex_support(c, q, ev[int(i)]) for c, q, i in zip(v.claim, v.question, v.id)]
    lex = v["lex"].fillna(v["entail_ret"])
    v["sup_nli"] = v["entail_ret"]
    v["sup_mean"] = (v["entail_ret"] + lex) / 2
    v["sup_max"] = np.maximum(v["entail_ret"], lex)
    return v