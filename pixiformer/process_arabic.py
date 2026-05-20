import re
import os
import requests
import pyarrow.parquet as pq
import pandas as pd
from huggingface_hub import HfApi, login

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATASET = "ClusterlabAi/101_billion_arabic_words_dataset"
OUTPUT_REPO = "nomeda-lab/msa-training"
NUM_FILES = 100
RAW_DIR = "data"
FINAL_PARQUET = "msa_training_clean.parquet"

login(HF_TOKEN, add_to_git_credential=False)

TASHKEEL = re.compile(r'[\u064B-\u065F\u0670\u06D6-\u06ED\u08D0-\u08E3]')
TATWEEL = re.compile(r'[\u0640]')
LINKS = re.compile(r'https?://\S+|www\.\S+')
LATIN = re.compile(r'[a-zA-Z]')
EASTERN_ARABIC = str.maketrans('٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹', '01234567890123456789')

def clean_text(text):
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    text = LINKS.sub('', text)
    text = LATIN.sub('', text)
    text = TASHKEEL.sub('', text)
    text = TATWEEL.sub('', text)
    text = re.sub(r'[إأآٱ]', 'ا', text)
    text = re.sub(r'[ى]', 'ي', text)
    text = re.sub(r'[ة]', 'ه', text)
    text = text.translate(EASTERN_ARABIC)
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return None
    return text

os.makedirs(RAW_DIR, exist_ok=True)

api = HfApi()
base_url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/data/"

all_seen = set()
all_cleaned = []

for file_idx in range(NUM_FILES):
    fname = f"train-{file_idx:05d}-of-00470.parquet"
    fpath = os.path.join(RAW_DIR, fname)

    url = base_url + fname
    r = requests.get(url, timeout=600)
    with open(fpath, 'wb') as f:
        f.write(r.content)

    table = pq.read_table(fpath, columns=['text'])
    texts = table.column('text').to_pylist()
    os.remove(fpath)

    file_new = 0
    for t in texts:
        cleaned = clean_text(t)
        if cleaned and cleaned not in all_seen:
            all_seen.add(cleaned)
            all_cleaned.append(cleaned)
            file_new += 1

    print(f"  {fname}: {len(texts)} rows → +{file_new} new (total unique: {len(all_seen)})")

    if (file_idx + 1) % 10 == 0:
        batch_df = pd.DataFrame({'text': all_cleaned})
        batch_df.to_parquet(FINAL_PARQUET, index=False)
        print(f"  >>> Saved progress ({len(all_cleaned)} rows) to {FINAL_PARQUET}")

print(f"\nDone! Total unique: {len(all_cleaned)}")

api.create_repo(repo_id=OUTPUT_REPO, repo_type="dataset", exist_ok=True)
api.upload_file(
    path_or_fileobj=FINAL_PARQUET,
    path_in_repo="data/msa_training_clean.parquet",
    repo_id=OUTPUT_REPO,
    repo_type="dataset",
)
print(f"Uploaded to https://huggingface.co/datasets/{OUTPUT_REPO}")
