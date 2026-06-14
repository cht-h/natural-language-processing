import pandas as pd
from sacrebleu.metrics import BLEU, CHRF
from model import My_Translator_Model

model = My_Translator_Model()
model._load_model()

df = pd.read_csv("data/dev.csv").dropna(subset=["transliteration", "translation"])
df = df.head(100)

predictions = []
references = []

for i, row in df.iterrows():
    inputs = model.tokenizer(
        str(row["transliteration"]),
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(model.device)

    import torch
    with torch.no_grad():
        output_ids = model.model.generate(
            **inputs,
            num_beams=4,
            max_new_tokens=256,
        )

    pred = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    predictions.append(pred)
    references.append(str(row["translation"]))

    if (i+1) % 10 == 0:
        print(f"Progress: {i+1}/100")

bleu = BLEU(tokenize="13a")
chrf = CHRF(word_order=2)

print(f"\nBLEU (no norm):   {bleu.corpus_score(predictions, [references]).score:.2f}")
print(f"chrF++ (no norm): {chrf.corpus_score(predictions, [references]).score:.2f}")