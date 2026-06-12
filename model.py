import re
import logging
import unicodedata
from pathlib import Path
from typing import Iterator

import fire
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import evaluate

#logging setup
LOG_PATH = Path("data/logs/log_file.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def normalize_akkadian(text: str) -> str:
    """Normalize Akkadian transliteration text."""
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\{.*?\}", "", text)

    #subscript digits to regular digits
    subs = {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
            "5": "5", "6": "6", "7": "7", "8": "8", "9": "9"}
    result = []
    for ch in text:
        name = unicodedata.name(ch, "")
        if "SUBSCRIPT DIGIT" in name or "SUPERSCRIPT DIGIT" in name:
            result.append(str(unicodedata.digit(ch)))
        else:
            result.append(ch)
    text = "".join(result)

    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


class AkkadianDataset(Dataset):
    def __init__(self, df, tokenizer, max_input_len=512, max_target_len=256):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

        df = df.dropna(subset=["transliteration", "translation"]).reset_index(drop=True)
        self.sources = df["transliteration"].tolist()
        self.targets = df["translation"].tolist()
        logger.info(f"Dataset loaded: {len(self.sources)} samples")

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        src = normalize_akkadian(self.sources[idx])
        tgt = self.targets[idx]

        model_inputs = self.tokenizer(
            src,
            max_length=self.max_input_len,
            truncation=True,
            padding=False,
        )
        labels = self.tokenizer(
            text_target=tgt,
            max_length=self.max_target_len,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


class My_Translator_Model:
    MODEL_NAME = "google/byt5-small"
    MODEL_DIR = Path("./model")
    RESULTS_DIR = Path("./data")

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

    def _load_model(self):
        if self.model is not None:
            return
        model_path = str(self.MODEL_DIR) if self.MODEL_DIR.exists() else self.MODEL_NAME
        logger.info(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, weights_only=False
        )
        self.model.to(self.device)
        logger.info("Model loaded successfully")

    def train(self, dataset_path: str) -> None:
        """Fine-tune ByT5 on Akkadian->English corpus. Saves to ./model/"""
        print("TRAIN STARTED", flush=True)
        logger.info(f"Starting training: {dataset_path}")

        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded {len(df)} rows, columns: {df.columns.tolist()}")

        dev_df = df.tail(500).reset_index(drop=True)
        train_df = df.iloc[:-500].reset_index(drop=True)
        logger.info(f"Train: {len(train_df)}, Dev: {len(dev_df)}")
        dev_df.to_csv("data/dev.csv", index=False)

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        train_dataset = AkkadianDataset(train_df, self.tokenizer)
        dev_dataset = AkkadianDataset(dev_df, self.tokenizer)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.MODEL_NAME, weights_only=False
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.MODEL_DIR),
            num_train_epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=200,
            weight_decay=0.01,
            learning_rate=5e-4,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            predict_with_generate=True,
            generation_max_length=256,
            fp16=torch.cuda.is_available(),
            logging_dir="data/logs",
            logging_steps=50,
            report_to="none",
            save_total_limit=2,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        logger.info("Starting training...")
        trainer.train()

        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(self.MODEL_DIR))
        self.tokenizer.save_pretrained(str(self.MODEL_DIR))
        logger.info(f"Model saved to {self.MODEL_DIR}")

    def predict(self, text: str, stream: bool = False, beam_size: int = 4):
        """Translate Akkadian to English. stream=True yields tokens."""
        self._load_model()

        text = normalize_akkadian(text)
        logger.info(f"Translating: {text[:80]}")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                num_beams=beam_size,
                max_new_tokens=256,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
            )

        translation = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logger.info(f"Translation: {translation[:80]}")

        if stream:
            for word in translation.split():
                yield word + " "
        else:
            return translation

    def predict_file(self, dataset_path: str) -> None:
        """Translate all rows, save to ./data/results.csv"""
        self._load_model()
        logger.info(f"Predicting file: {dataset_path}")

        df = pd.read_csv(dataset_path)
        translations = []

        for i, row in df.iterrows():
            text = row.get("transliteration", row.get("text", ""))
            translation = self.predict(text, stream=False)
            translations.append(translation)
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(df)}")

        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame({
            "id": df["id"] if "id" in df.columns else range(len(df)),
            "translation": translations,
        })
        results_df.to_csv(self.RESULTS_DIR / "results.csv", index=False)
        logger.info("Results saved to data/results.csv")

    def evaluate_dev(self, dev_path: str = "data/dev.csv") -> dict:
        """Evaluate on dev set, report BLEU and chrF++."""
        self._load_model()

        df = pd.read_csv(dev_path).dropna(subset=["transliteration", "translation"])
        predictions, references = [], []

        for i, row in df.iterrows():
            pred = self.predict(row["transliteration"], stream=False)
            predictions.append(pred)
            references.append(row["translation"])
            if (i + 1) % 20 == 0:
                logger.info(f"Eval progress: {i+1}/{len(df)}")

        bleu = evaluate.load("sacrebleu")
        bleu_score = bleu.compute(
            predictions=predictions,
            references=[[r] for r in references],
            tokenize="13a",
        )

        chrf = evaluate.load("chrf")
        chrf_score = chrf.compute(
            predictions=predictions,
            references=[[r] for r in references],
            word_order=2,
        )

        results = {
            "bleu": round(bleu_score["score"], 2),
            "chrf++": round(chrf_score["score"], 2),
            "num_samples": len(predictions),
        }
        logger.info(f"Evaluation results: {results}")
        return results


if __name__ == "__main__":
    fire.Fire(My_Translator_Model())