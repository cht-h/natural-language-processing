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
    result = model.predict(str(row["transliteration"]), stream=False)
    pred = result if isinstance(result, str) else "".join(result)
    predictions.append(pred)
    references.append(str(row["translation"]))
    if (i+1) % 10 == 0:
        print(f"Progress: {i+1}/100")

bleu = BLEU(tokenize="13a")
chrf = CHRF(word_order=2)

bleu_score = bleu.corpus_score(predictions, [references])
chrf_score = chrf.corpus_score(predictions, [references])

print(f"\nBLEU:   {bleu_score.score:.2f}")
print(f"chrF++: {chrf_score.score:.2f}")