import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

NLI_MODEL = os.getenv("NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

_tok = AutoTokenizer.from_pretrained(NLI_MODEL)
_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).eval()


def _find(prefix):
    for i, label in _model.config.id2label.items():
        if str(label).lower().startswith(prefix):
            return int(i)
    raise ValueError(f"No '{prefix}' label in {_model.config.id2label}")


E, N, C = _find("entail"), _find("neutral"), _find("contra")
print("NLI model:", NLI_MODEL, "| labels:", _model.config.id2label)


@torch.no_grad()
def nli_scores(premises, hypotheses, batch_size=8):
    out = []
    for i in tqdm(range(0, len(premises), batch_size)):
        enc = _tok(
            premises[i:i + batch_size],
            hypotheses[i:i + batch_size],
            truncation="only_first",
            max_length=512,
            padding=True,
            return_tensors="pt",
        )
        probs = torch.softmax(_model(**enc).logits, dim=-1).numpy()
        out.extend(
            {"entail": float(r[E]), "neutral": float(r[N]), "contra": float(r[C])}
            for r in probs
        )
    return out