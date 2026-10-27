"""LoRA fine-tuning of the intent LLM (paper: Llama-3.1-8B-Instruct, rank 16, alpha 32, lr 2e-4, 3 epochs).

Requires the optional extra (``pip install -e ".[llm]"``), access to the model
weights, and in practice a GPU with enough memory for an 8B model.
NOT executed during repository validation (see SMOKE_TEST_REPORT.md).

    python scripts/finetune_intent_lora.py --corpus data/corpus/train.jsonl \
        --model meta-llama/Llama-3.1-8B-Instruct --out checkpoints/intent_lora

Corpus format: JSON lines with "text", "context" (SessionContext) and
"constraint" (gold ChargingConstraint), as produced by scripts/intent_corpus.py.
The loss is computed on the target JSON tokens only.
"""

from __future__ import annotations

import argparse
import random

import _bootstrap  # noqa: F401
from solace.intent import ChargingConstraint, SessionContext
from solace.intent.corpus import load_jsonl
from solace.intent.parsers import HFParser, target_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)  # paper
    ap.add_argument("--alpha", type=int, default=32)  # paper
    ap.add_argument("--dropout", type=float, default=0.05)  # paper (original Table III)
    ap.add_argument("--lr", type=float, default=2e-4)  # paper
    ap.add_argument("--epochs", type=int, default=3)  # paper
    ap.add_argument("--batch-size", type=int, default=16)  # paper (original Table III)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("install the optional extra first: pip install -e '.[llm]'") from exc

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                                             task_type="CAUSAL_LM"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    examples = []
    for rec in load_jsonl(args.corpus):
        ctx = SessionContext.model_validate(rec["context"])
        prompt_ids = tok.apply_chat_template(HFParser.build_messages(rec["text"], ctx), add_generation_prompt=True)
        target_ids = tok(target_json(ChargingConstraint.model_validate(rec["constraint"])),
                         add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        examples.append((prompt_ids + target_ids, [-100] * len(prompt_ids) + target_ids))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    for epoch in range(args.epochs):
        random.shuffle(examples)
        total = 0.0
        for i in range(0, len(examples), args.batch_size):
            batch = examples[i:i + args.batch_size]
            width = max(len(x) for x, _ in batch)
            ids = torch.full((len(batch), width), tok.pad_token_id)
            labels = torch.full((len(batch), width), -100)
            attn = torch.zeros((len(batch), width), dtype=torch.long)
            for j, (x, y) in enumerate(batch):
                ids[j, : len(x)] = torch.tensor(x)
                labels[j, : len(y)] = torch.tensor(y)
                attn[j, : len(x)] = 1
            out = model(input_ids=ids.to(device), attention_mask=attn.to(device), labels=labels.to(device))
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            total += out.loss.item() * len(batch)
        print(f"epoch {epoch + 1}: loss {total / max(len(examples), 1):.4f}")
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
