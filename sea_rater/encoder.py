"""
Shared frozen-encoder embedding logic (pipeline.md Section 4).

Used both to embed the human-labeled splits for rater training and,
later, to score the candidate corpus (Section 5) with the same backbone.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-large"
DEFAULT_MAX_LENGTH = 512
# E5 models are trained with an instruction prefix distinguishing queries
# from documents; we only ever embed documents here.
DOCUMENT_PREFIX = "passage: "


def load_encoder(model_name=DEFAULT_MODEL_NAME, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return tokenizer, model


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def embed_texts(
    texts,
    tokenizer,
    model,
    device="cuda",
    batch_size=32,
    max_length=DEFAULT_MAX_LENGTH,
    prefix=DOCUMENT_PREFIX,
    show_progress=True,
):
    """Embed a list of raw document strings with the frozen E5 backbone.

    Returns a float32 tensor of shape (len(texts), hidden_size), L2-normalized
    (E5 embeddings are trained for cosine-similarity use, so we keep them
    normalized; the MLP heads can still learn arbitrary functions of them).
    """
    embeddings = []
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="embedding", total=(len(texts) + batch_size - 1) // batch_size)

    for start in iterator:
        batch = [prefix + t for t in texts[start : start + batch_size]]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        outputs = model(**encoded)
        pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=-1)
        embeddings.append(pooled.cpu())

    return torch.cat(embeddings, dim=0)
