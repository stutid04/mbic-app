# trainbias.py — M2 Air (8GB) friendly: freeze encoder, short seq, small batch
import os, random, sys, numpy as np, pandas as pd, torch

# MPS safety/fallbacks
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# If you still OOM, uncomment next line to lift Apple’s cap (use with care):
# os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

import evaluate
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)

print("Python:", sys.version)
print("Torch :", torch.__version__, "| MPS available:",
      getattr(torch.backends, "mps").is_available() if hasattr(torch.backends, "mps") else False)

# ---------------- Data ----------------
XLSX_PATH = "labeled_dataset.xlsx"
df = pd.read_excel(XLSX_PATH)
df = df[["sentence","Label_bias"]].dropna()
label_map = {"Non-biased": 0, "Biased": 1}
df = df[df["Label_bias"].isin(label_map)]
df["label"] = df["Label_bias"].map(label_map)
df["sentence"] = df["sentence"].astype(str).str.replace(r"\s+"," ", regex=True).str.strip()

train_df, test_df = train_test_split(
    df[["sentence","label"]], test_size=0.2, random_state=SEED, stratify=df["label"]
)
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
test_ds  = Dataset.from_pandas(test_df.reset_index(drop=True))
print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

# ------------- Model (small + frozen) -------------
MODEL_NAME = "distilbert-base-uncased"   # lighter than DistilRoBERTa for 8GB
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Freeze encoder (massive memory savings; trains only the classifier head)
for p in model.base_model.parameters():
    p.requires_grad = False

# Optional: gradient checkpointing (little effect when frozen; harmless)
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

MAX_LEN = 128
def tokenize(batch):
    return tok(batch["sentence"], truncation=True, max_length=MAX_LEN)

train_ds = train_ds.map(tokenize, batched=True, remove_columns=["sentence"])
test_ds  = test_ds.map(tokenize, batched=True, remove_columns=["sentence"])
collator = DataCollatorWithPadding(tokenizer=tok)

# -------------- Metrics --------------
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        "f1_weighted": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }

# -------------- Training args --------------
OUT_DIR = "out/distilbert-mbic-binary"
common = dict(
    output_dir=OUT_DIR,
    per_device_train_batch_size=8,      # small batch fits on 8GB when frozen
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,      # effective batch = 8
    learning_rate=5e-5,                 # slightly higher LR when head-only
    num_train_epochs=4,
    weight_decay=0.0,
    logging_steps=50,
    save_total_limit=2,
    seed=SEED,
    bf16=False,                          # Apple Silicon supports bf16 on MPS
    report_to="none",
)

try:
    args = TrainingArguments(
        **common,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        gradient_checkpointing=True,
    )
except TypeError:
    # Older transformers fallback (no evaluation_strategy param)
    args = TrainingArguments(
        **common,
        do_eval=True,
        logging_dir=os.path.join(OUT_DIR, "logs"),
        save_steps=500,
        eval_steps=500,
    )

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    processing_class=tok,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# Resume if a checkpoint exists
resume = any(name.startswith("checkpoint-") for name in os.listdir(OUT_DIR)) if os.path.exists(OUT_DIR) else False
print("Starting training...", "(resuming)" if resume else "")
try:
    if hasattr(torch, "mps"): torch.mps.empty_cache()
except Exception: pass

trainer.train(resume_from_checkpoint=resume)

# -------------- Save + sanity check --------------
BEST = os.path.join(OUT_DIR, "best")
trainer.save_model(BEST)
tok.save_pretrained(BEST)
print("Saved model to:", BEST)
print("Eval:", trainer.evaluate())

from transformers import pipeline
clf = pipeline("text-classification", model=BEST, tokenizer=tok, return_all_scores=True)
print(clf("The policy drew widespread criticism from several groups."))