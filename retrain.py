"""
VoxIQ — DistilBERT Retraining Script
Run standalone or as a SageMaker entry point.

Usage:
    python retrain.py

Expects AWS credentials with s3:GetObject / s3:PutObject on
s3://operationcapstone-models/
"""

import io
import json
import os
import tempfile
from datetime import datetime, timezone

import boto3
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)
import torch
from torch.utils.data import Dataset

BUCKET     = "operationcapstone-models"
TRAIN_PFX  = "training_data/"
MODEL_PFX  = "distilbert_finetuned/"
VERSION_KEY = "model_versions/log.json"
MODEL_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "tokenizer.json",
]

s3 = boto3.client("s3")


# ── Data ──────────────────────────────────────────────────────────────────────

def download_training_data() -> pd.DataFrame:
    paginator = s3.get_paginator("list_objects_v2")
    frames = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=TRAIN_PFX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".csv"):
                continue
            body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            frames.append(pd.read_csv(io.BytesIO(body)))
    if not frames:
        raise RuntimeError("No training CSVs found under training_data/")
    df = pd.concat(frames, ignore_index=True)
    print(f"Downloaded {len(df):,} rows from {len(frames)} file(s).")
    return df


def prepare_dataset(df: pd.DataFrame):
    df = df[["text", "predicted_sentiment"]].dropna()
    df = df[df["predicted_sentiment"].isin(["positive", "negative"])]
    df["label"] = (df["predicted_sentiment"] == "positive").astype(int)
    train_df, eval_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"Train: {len(train_df):,}  Eval: {len(eval_df):,}")
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


# ── Model ─────────────────────────────────────────────────────────────────────

def download_base_model(tmp_dir: str):
    for fname in MODEL_FILES:
        dest = os.path.join(tmp_dir, fname)
        if not os.path.exists(dest):
            s3.download_file(BUCKET, MODEL_PFX + fname, dest)
    print(f"Base model downloaded to {tmp_dir}")


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        enc = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels         = torch.tensor(labels.tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}


def train(train_df, eval_df, model_dir: str, output_dir: str) -> float:
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model     = DistilBertForSequenceClassification.from_pretrained(
        model_dir, num_labels=2
    )

    train_ds = ReviewDataset(train_df["text"], train_df["label"], tokenizer)
    eval_ds  = ReviewDataset(eval_df["text"],  eval_df["label"],  tokenizer)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    final_metrics = trainer.evaluate()
    accuracy = final_metrics["eval_accuracy"]
    print(f"Final eval accuracy: {accuracy:.4f}")

    # Save best model to output_dir for upload
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return accuracy


# ── S3 upload ─────────────────────────────────────────────────────────────────

def upload_model_to_s3(model_dir: str):
    for fname in MODEL_FILES:
        src = os.path.join(model_dir, fname)
        if os.path.exists(src):
            s3.upload_file(src, BUCKET, MODEL_PFX + fname)
            print(f"  Uploaded {fname}")
    print("Model upload complete.")


# ── Version log ───────────────────────────────────────────────────────────────

def write_version_log(rows_trained: int, accuracy: float) -> int:
    log = []
    try:
        body = s3.get_object(Bucket=BUCKET, Key=VERSION_KEY)["Body"].read()
        log  = json.loads(body)
    except s3.exceptions.NoSuchKey:
        pass
    except Exception as e:
        print(f"Warning: could not load existing version log: {e}")

    version = (log[-1]["version"] + 1) if log else 1
    log.append({
        "version":      version,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "rows_trained": rows_trained,
        "accuracy":     round(accuracy, 6),
    })
    s3.put_object(
        Bucket=BUCKET,
        Key=VERSION_KEY,
        Body=json.dumps(log, indent=2).encode(),
        ContentType="application/json",
    )
    print(f"Version log updated: v{version}")
    return version


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    df = download_training_data()
    train_df, eval_df = prepare_dataset(df)
    rows_trained = len(train_df) + len(eval_df)

    with tempfile.TemporaryDirectory() as model_dir:
        download_base_model(model_dir)
        with tempfile.TemporaryDirectory() as output_dir:
            accuracy = train(train_df, eval_df, model_dir, output_dir)
            upload_model_to_s3(output_dir)

    version = write_version_log(rows_trained, accuracy)
    print(f"\nRetraining complete: v{version} · {rows_trained:,} rows · {accuracy*100:.2f}% accuracy")


if __name__ == "__main__":
    main()
