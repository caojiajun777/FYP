"""
Task2 route-B: Qwen2.5-VL-7B + LoRA training for multilabel schematic function classification.

This script keeps your existing gold dataset format and reports metrics that are comparable
with ResNet/ViT baselines.

Usage examples:
  python train_task2_qwen_vl_lora.py
  python train_task2_qwen_vl_lora.py --epochs 4 --batch-size 1 --grad-accum 8
  python train_task2_qwen_vl_lora.py --model-name Qwen/Qwen2.5-VL-7B-Instruct

Outputs:
  task2_qwen_vl_lora/
    best_adapter/                 # best LoRA adapter + processor
        last_adapter/                 # latest LoRA adapter for resume
        resume_state.pt               # optimizer/scheduler/epoch state for resume
    training_log.json             # per-epoch loss and val metrics
    test_metrics.json             # final test metrics
    test_predictions.json         # per-sample predictions
    split_files.json              # train/val/test file lists
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, f1_score, hamming_loss, multilabel_confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from peft import LoraConfig, PeftModel, get_peft_model


CLASSES = ["power", "communication", "interface", "control", "signal", "eval_board"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}

SYSTEM_PROMPT = (
    "You are an expert EDA schematic function classifier. "
    "Classify the schematic image into one or more labels from this exact label set: "
    "power, communication, interface, control, signal, eval_board."
)

USER_PROMPT = (
    "Return ONLY a JSON array using the exact label names. "
    "No explanation, no extra text. "
    "Example: [\"power\", \"control\"]"
)


@dataclass
class Record:
    filename: str
    image_path: str
    gold_labels: List[str]


class GoldRecordDataset(Dataset):
    def __init__(self, records: List[Record]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Record:
        return self.records[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def labels_to_json_text(labels: List[str]) -> str:
    ordered = [c for c in CLASSES if c in labels]
    return json.dumps(ordered, ensure_ascii=True)


def build_messages(image_path: str, answer_text: str = None) -> List[Dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]
    if answer_text is not None:
        messages.append({"role": "assistant", "content": answer_text})
    return messages


def parse_predicted_labels(raw_text: str) -> List[str]:
    text = (raw_text or "").strip()
    if not text:
        return []

    # Try direct JSON parse first.
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            cleaned = []
            for item in obj:
                s = str(item).strip().lower()
                if s in CLASS_TO_IDX and s not in cleaned:
                    cleaned.append(s)
            return cleaned
    except Exception:
        pass

    # Try extracting first [ ... ] chunk.
    if "[" in text and "]" in text:
        chunk = text[text.find("[") : text.find("]") + 1]
        try:
            obj = json.loads(chunk)
            if isinstance(obj, list):
                cleaned = []
                for item in obj:
                    s = str(item).strip().lower()
                    if s in CLASS_TO_IDX and s not in cleaned:
                        cleaned.append(s)
                return cleaned
        except Exception:
            pass

    lowered = text.lower()
    aliases = {
        "power": ["power", "power_management", "power management"],
        "communication": ["communication", "comm"],
        "interface": ["interface", "io", "i/o"],
        "control": ["control", "controller"],
        "signal": ["signal", "signal_processing", "signal processing"],
        "eval_board": ["eval_board", "evaluation_board", "evaluation board", "dev board", "development board"],
    }

    found = []
    for cls_name in CLASSES:
        for token in aliases[cls_name]:
            if token in lowered:
                found.append(cls_name)
                break
    return found


def records_to_multihot(records: List[Record]) -> np.ndarray:
    y = np.zeros((len(records), len(CLASSES)), dtype=np.int32)
    for i, rec in enumerate(records):
        for label in rec.gold_labels:
            if label in CLASS_TO_IDX:
                y[i, CLASS_TO_IDX[label]] = 1
    return y


def load_gold_records(gold_file: str) -> List[Record]:
    with open(gold_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for r in raw:
        img = r.get("image_path", "")
        labels = [l for l in r.get("gold_labels", []) if l in CLASS_TO_IDX]
        if os.path.exists(img) and labels:
            records.append(Record(filename=r["filename"], image_path=img, gold_labels=labels))

    return records


def split_records(
    records: List[Record],
    seed: int,
    test_split_file: str,
    val_ratio: float,
) -> Dict[str, List[Record]]:
    by_name = {r.filename: r for r in records}

    if os.path.exists(test_split_file):
        with open(test_split_file, "r", encoding="utf-8") as f:
            test_names = set(json.load(f))
        test = [by_name[n] for n in test_names if n in by_name]
        remain = [r for r in records if r.filename not in test_names]
    else:
        rng = random.Random(seed)
        shuffled = records[:]
        rng.shuffle(shuffled)
        n_test = max(1, int(round(0.1 * len(shuffled))))
        test = shuffled[:n_test]
        remain = shuffled[n_test:]

    rng = random.Random(seed)
    rng.shuffle(remain)
    n_val = max(1, int(round(val_ratio * len(remain))))
    val = remain[:n_val]
    train = remain[n_val:]

    return {"train": train, "val": val, "test": test}


def choose_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def safe_torch_save(payload: Dict, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, save_path)


def build_train_collate_fn(processor: AutoProcessor, max_length: int):
    pad_id = processor.tokenizer.pad_token_id

    def collate(batch: List[Record]) -> Dict[str, torch.Tensor]:
        images = []
        texts_full = []

        for rec in batch:
            with Image.open(rec.image_path) as im:
                image = im.convert("RGB")

            answer_text = labels_to_json_text(rec.gold_labels)
            full_messages = build_messages(rec.image_path, answer_text)

            text_full = processor.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )

            images.append(image)
            texts_full.append(text_full)

        model_inputs = processor(
            text=texts_full,
            images=images,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        labels = model_inputs["input_ids"].clone()
        if pad_id is not None:
            labels[labels == pad_id] = -100

        model_inputs["labels"] = labels
        return model_inputs

    return collate


def evaluate_generative(
    model,
    processor: AutoProcessor,
    records: List[Record],
    device: torch.device,
    max_new_tokens: int,
) -> Dict:
    model.eval()
    y_true = records_to_multihot(records)
    y_pred = np.zeros_like(y_true)
    samples = []

    for i, rec in enumerate(tqdm(records, desc="Evaluating", leave=False)):
        prompt_messages = build_messages(rec.image_path, None)
        prompt_text = processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        with Image.open(rec.image_path) as im:
            image = im.convert("RGB")

        inputs = processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                use_cache=True,
            )

        input_len = int(inputs["input_ids"].shape[1])
        new_tokens = output_ids[:, input_len:]
        raw_text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        pred_labels = parse_predicted_labels(raw_text)

        for label in pred_labels:
            y_pred[i, CLASS_TO_IDX[label]] = 1

        samples.append(
            {
                "filename": rec.filename,
                "gold": [c for c in CLASSES if y_true[i, CLASS_TO_IDX[c]] == 1],
                "pred": [c for c in CLASSES if y_pred[i, CLASS_TO_IDX[c]] == 1],
                "raw_output": raw_text,
            }
        )

    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    hamming = float(hamming_loss(y_true, y_pred))

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    mcm = multilabel_confusion_matrix(y_true, y_pred)
    confusion = {}
    for i, cls_name in enumerate(CLASSES):
        tn, fp, fn, tp = mcm[i].ravel().tolist()
        confusion[cls_name] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

    return {
        "macro_f1": macro,
        "micro_f1": micro,
        "hamming": hamming,
        "classification_report": report,
        "confusion": confusion,
        "n_samples": len(records),
        "samples": samples,
    }


def print_split_stats(name: str, records: List[Record]) -> None:
    y = records_to_multihot(records)
    counts = y.sum(axis=0)
    print(f"[{name}] n={len(records)}")
    for i, cls_name in enumerate(CLASSES):
        print(f"  - {cls_name:<14} {int(counts[i]):>4}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-file", default="task2_function_prediction/gold_subset.json")
    parser.add_argument("--test-split-file", default="task2_gold_models/test_split.json")
    parser.add_argument("--output-dir", default="task2_qwen_vl_lora")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-7B-Instruct")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.1)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--min-pixels", type=int, default=200704, help="Min image pixels budget for Qwen-VL processor")
    parser.add_argument("--max-pixels", type=int, default=1003520, help="Max image pixels budget for Qwen-VL processor")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume training from output-dir checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = choose_dtype()

    out_dir = Path(args.output_dir)
    best_dir = out_dir / "best_adapter"
    last_adapter_dir = out_dir / "last_adapter"
    resume_state_path = out_dir / "resume_state.pt"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] device={device}, dtype={dtype}")
    print(f"[INFO] loading records from {args.gold_file}")
    records = load_gold_records(args.gold_file)
    if len(records) < 50:
        raise RuntimeError(f"Too few valid records: {len(records)}")

    split = split_records(
        records=records,
        seed=args.seed,
        test_split_file=args.test_split_file,
        val_ratio=args.val_ratio,
    )

    print_split_stats("train", split["train"])
    print_split_stats("val", split["val"])
    print_split_stats("test", split["test"])

    with open(out_dir / "split_files.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train": [r.filename for r in split["train"]],
                "val": [r.filename for r in split["val"]],
                "test": [r.filename for r in split["test"]],
            },
            f,
            indent=2,
        )

    can_resume = args.resume and last_adapter_dir.exists() and resume_state_path.exists()
    if args.resume and not can_resume:
        print(
            "[WARN] --resume requested but checkpoint files are missing. "
            "Will start fresh training and create resume files after epoch 1."
        )

    processor_source = str(last_adapter_dir) if can_resume else args.model_name
    print(f"[INFO] loading model and processor: {args.model_name}")
    if args.resume:
        print(f"[INFO] resume mode: processor source = {processor_source}")
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    if hasattr(processor, "image_processor"):
        # Limit visual token explosion on very large schematic images.
        processor.image_processor.min_pixels = args.min_pixels
        processor.image_processor.max_pixels = args.max_pixels

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    base_model.to(device)

    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
    if hasattr(base_model, "config"):
        base_model.config.use_cache = False

    # Freeze base model, then attach LoRA adapters.
    for p in base_model.parameters():
        p.requires_grad = False

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if can_resume:
        if not last_adapter_dir.exists() or not resume_state_path.exists():
            raise FileNotFoundError(
                f"Resume requested but checkpoint files are missing: {last_adapter_dir} and/or {resume_state_path}"
            )
        model = PeftModel.from_pretrained(base_model, last_adapter_dir, is_trainable=True)
        print(f"[INFO] loaded resume adapter from: {last_adapter_dir}")
    else:
        model = get_peft_model(base_model, lora_cfg)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.print_trainable_parameters()

    train_ds = GoldRecordDataset(split["train"])
    train_collate = build_train_collate_fn(processor, max_length=args.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_collate,
    )

    total_updates_per_epoch = max(1, (len(train_loader) + args.grad_accum - 1) // args.grad_accum)
    total_updates = total_updates_per_epoch * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_updates - warmup_steps),
    )

    training_log = []
    best_val_macro = -1.0
    global_step = 0
    start_epoch = 1

    if can_resume:
        resume_state = torch.load(resume_state_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])

        training_log = resume_state.get("training_log", [])
        best_val_macro = float(resume_state.get("best_val_macro", -1.0))
        global_step = int(resume_state.get("global_step", 0))
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        print(
            f"[INFO] resumed state: last_epoch={start_epoch - 1}, "
            f"best_val_macro={best_val_macro:.4f}, global_step={global_step}"
        )

    print("[INFO] start training")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar, start=1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch, labels=labels)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            running_loss += float(loss.item() * args.grad_accum)

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                if global_step > warmup_steps:
                    scheduler.step()

            avg_loss = running_loss / step
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # Flush remaining grads if dataloader length is not divisible by grad_accum.
        if len(train_loader) % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step > warmup_steps:
                scheduler.step()

        train_loss = running_loss / max(1, len(train_loader))

        val_metrics = evaluate_generative(
            model=model,
            processor=processor,
            records=split["val"],
            device=device,
            max_new_tokens=args.max_new_tokens,
        )

        log_item = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_micro_f1": val_metrics["micro_f1"],
            "val_hamming": val_metrics["hamming"],
        }
        training_log.append(log_item)

        print(
            f"[VAL] epoch={epoch} loss={train_loss:.4f} "
            f"macro_f1={val_metrics['macro_f1']:.4f} micro_f1={val_metrics['micro_f1']:.4f} "
            f"hamming={val_metrics['hamming']:.4f}"
        )

        with open(out_dir / "training_log.json", "w", encoding="utf-8") as f:
            json.dump(training_log, f, indent=2)

        if val_metrics["macro_f1"] > best_val_macro:
            best_val_macro = val_metrics["macro_f1"]
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"[INFO] saved best adapter to: {best_dir}")

        # Save resumable checkpoint every epoch.
        last_adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(last_adapter_dir)
        processor.save_pretrained(last_adapter_dir)
        safe_torch_save(
            {
                "epoch": epoch,
                "best_val_macro": best_val_macro,
                "global_step": global_step,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "training_log": training_log,
                "args": vars(args),
            },
            resume_state_path,
        )
        print(f"[INFO] saved resume checkpoint: epoch={epoch}")

    if start_epoch > args.epochs:
        print(
            f"[INFO] resume checkpoint already reached target epochs: "
            f"start_epoch={start_epoch}, target={args.epochs}. Skipping training loop."
        )

    print(f"[INFO] best val macro_f1 = {best_val_macro:.4f}")

    if not best_dir.exists():
        raise RuntimeError("Best adapter not found; training did not save any checkpoint.")

    print("[INFO] loading best adapter for final test")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    base_model.to(device)
    model = PeftModel.from_pretrained(base_model, best_dir)

    test_metrics = evaluate_generative(
        model=model,
        processor=processor,
        records=split["test"],
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    to_save_metrics = {
        "model": args.model_name,
        "macro_f1": test_metrics["macro_f1"],
        "micro_f1": test_metrics["micro_f1"],
        "hamming": test_metrics["hamming"],
        "classes": CLASSES,
        "n_test": test_metrics["n_samples"],
        "confusion": test_metrics["confusion"],
        "classification_report": test_metrics["classification_report"],
    }

    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(to_save_metrics, f, indent=2)

    with open(out_dir / "test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics["samples"], f, indent=2, ensure_ascii=False)

    print("\n[TEST] metrics")
    print(json.dumps(to_save_metrics, indent=2))
    print(f"[DONE] outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()
