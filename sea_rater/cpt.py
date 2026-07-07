"""
Shared continued-pretraining utilities for the proxy CPT search
(pipeline.md Section 9) and, later, the final CPT run (Section 12).
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

PROXY_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
FINAL_MODEL_NAME = "Qwen/Qwen2.5-1.5B"

# Qwen2's attention/MLP projection names -- the standard LoRA target set.
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_cpt_model_and_tokenizer(model_name=PROXY_MODEL_NAME, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.to(device)
    return tokenizer, model


def apply_lora(model, r=16, lora_alpha=32, lora_dropout=0.05, target_modules=None):
    """Wrap `model` for LoRA fine-tuning (pipeline.md Section 12: "Train a
    model: Qwen2.5-1.5B-CPT-BestWeighted with LoRa"). Freezes the base
    model and adds small trainable adapter matrices to the attention/MLP
    projections; only those adapter weights end up with requires_grad=True.
    """
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules or DEFAULT_LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def trainable_parameters(model):
    return (p for p in model.parameters() if p.requires_grad)


def pack_texts(texts, tokenizer, seq_length):
    """Concatenate tokenized documents (EOS-separated) into a stream, then
    chunk into fixed-length blocks of `seq_length` tokens, dropping the
    final incomplete block -- the standard packed-pretraining approach.

    Returns a tensor of shape (num_blocks, seq_length).
    """
    eos_id = tokenizer.eos_token_id
    all_ids = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_ids.extend(ids)
        all_ids.append(eos_id)

    num_blocks = len(all_ids) // seq_length
    usable = num_blocks * seq_length
    if num_blocks == 0:
        raise ValueError(
            f"Only {len(all_ids)} tokens available, less than one block of {seq_length}"
        )
    blocks = torch.tensor(all_ids[:usable], dtype=torch.long).view(num_blocks, seq_length)
    return blocks


def make_block_loader(blocks, batch_size, shuffle):
    dataset = TensorDataset(blocks)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_batches = 0
    for (batch,) in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=batch, labels=batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for (batch,) in loader:
        batch = batch.to(device)
        outputs = model(input_ids=batch, labels=batch)
        total_loss += outputs.loss.item()
        total_batches += 1
    return total_loss / max(total_batches, 1)
