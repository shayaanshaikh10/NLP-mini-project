"""
train_transformer.py - OPTIONAL: fine-tune XLM-RoBERTa on the Hinglish data.

This only runs when PyTorch is installed AND a CUDA GPU is available.
On a normal laptop it prints a message and exits without error.
No GPU? Use notebooks/colab_transformer_finetune.ipynb on free Google Colab.

Install the extra libraries first (see the optional part of requirements.txt):
    pip install torch transformers datasets accelerate

Run:  python -m src.train_transformer
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src import config
from src.utils import Timer, log_step, save_json

LABEL_TO_ID = {label: index for index, label in enumerate(config.LABELS)}


def gpu_is_available():
    """Return True only if torch is installed and can see a CUDA GPU."""
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed (optional). Skipping transformer training.")
        return False
    if not torch.cuda.is_available():
        print("No CUDA GPU found. Skipping transformer training.")
        return False
    return True


def load_as_hf_dataset(path, tokenizer):
    """Read a processed CSV and tokenize it into a Hugging Face Dataset."""
    from datasets import Dataset

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["labels"] = df[config.LABEL_COL].map(LABEL_TO_ID)
    dataset = Dataset.from_pandas(df[[config.TEXT_COL, "labels"]])

    def tokenize(batch):
        return tokenizer(batch[config.TEXT_COL], truncation=True,
                         max_length=config.TRANSFORMER_MAX_LEN, padding="max_length")

    return dataset.map(tokenize, batched=True)


def compute_metrics(eval_prediction):
    """Accuracy and macro-F1 for the Hugging Face Trainer."""
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, average="macro")}


def build_trainer(model, train_data, val_data):
    """Create a Trainer that keeps the checkpoint with the best validation macro-F1."""
    from transformers import Trainer, TrainingArguments

    arguments = TrainingArguments(
        output_dir=str(config.TRANSFORMER_DIR / "checkpoints"),
        num_train_epochs=config.TRANSFORMER_EPOCHS,
        learning_rate=config.TRANSFORMER_LR,
        per_device_train_batch_size=config.TRANSFORMER_BATCH_SIZE,
        per_device_eval_batch_size=config.TRANSFORMER_BATCH_SIZE * 2,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        fp16=True, seed=config.RANDOM_STATE, report_to="none")
    return Trainer(model=model, args=arguments, train_dataset=train_data,
                   eval_dataset=val_data, compute_metrics=compute_metrics)


def train_transformer():
    """Fine-tune, evaluate on the test set and save the model + metrics."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.TRANSFORMER_MODEL_NAME, num_labels=len(config.LABELS),
        id2label=dict(enumerate(config.LABELS)), label2id=LABEL_TO_ID)
    train_data = load_as_hf_dataset(config.TRAIN_PATH, tokenizer)
    val_data = load_as_hf_dataset(config.VAL_PATH, tokenizer)
    test_data = load_as_hf_dataset(config.TEST_PATH, tokenizer)

    trainer = build_trainer(model, train_data, val_data)
    with Timer() as timer:
        trainer.train()
    test_metrics = trainer.evaluate(test_data, metric_key_prefix="test")
    trainer.save_model(str(config.TRANSFORMER_DIR))
    tokenizer.save_pretrained(str(config.TRANSFORMER_DIR))
    save_json({"train_time_s": round(timer.seconds, 1), **test_metrics},
              config.TRANSFORMER_DIR / "metrics.json")
    return test_metrics


def main():
    """Train the transformer if possible, otherwise log that it was skipped."""
    if not gpu_is_available():
        log_step("Step 5 Transformer: skipped (no GPU). "
                 "Use notebooks/colab_transformer_finetune.ipynb on Colab instead.")
        return
    metrics = train_transformer()
    log_step(f"Step 5 Transformer: {config.TRANSFORMER_MODEL_NAME} test macro-F1 = "
             f"{metrics['test_macro_f1']:.4f}, saved to models/transformer/")


if __name__ == "__main__":
    main()
