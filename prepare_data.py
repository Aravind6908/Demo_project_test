import os
import json
import random

# ── CONFIG ───────────────────────────────────────────────────────────────────────
CASES_DIR     = "case_files/"       # folder of “case1.txt”, “case2.txt”, …
SUMMARIES_DIR = "summaries/"   # folder of “case1.txt”, “case2.txt”, …
OUTPUT_DIR    = "data/"
TRAIN_FILE    = os.path.join(OUTPUT_DIR, "train.json")
VAL_FILE      = os.path.join(OUTPUT_DIR, "val.json")
VAL_SPLIT     = 0.1                 # 10% of your data goes to validation
# ────────────────────────────────────────────────────────────────────────────────

def load_pairs():
    pairs = []
    for fname in os.listdir(CASES_DIR):
        if not fname.endswith(".txt"):
            continue
        base = os.path.splitext(fname)[0]
        case_path    = os.path.join(CASES_DIR, fname)
        summary_path = os.path.join(SUMMARIES_DIR, base + ".txt")
        if not os.path.exists(summary_path):
            print(f"⚠️  no summary for {base}, skipping")
            continue
        with open(case_path,    "r", encoding="utf-8") as fc, \
             open(summary_path, "r", encoding="utf-8") as fs:
            doc   = fc.read().strip()
            summ  = fs.read().strip()
            if doc and summ:
                pairs.append({
                    "document_text": doc,
                    "summary_text":  summ,
                    # optional: add a few_shot_prompt key here if you want
                })
    return pairs

def split_and_save(pairs):
    random.shuffle(pairs)
    split_idx = int(len(pairs) * (1 - VAL_SPLIT))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_data = {"data": pairs[:split_idx]}
    val_data   = {"data": pairs[split_idx:]}

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
    with open(VAL_FILE,   "w", encoding="utf-8") as f:
        json.dump(val_data,   f, indent=2)

    print(f"✅ Wrote {len(train_data['data'])} train examples to {TRAIN_FILE}")
    print(f"✅ Wrote {len(val_data  ['data'])} val   examples to {VAL_FILE}")

if __name__ == "__main__":
    pairs = load_pairs()
    print(f"ℹ️  Found {len(pairs)} total (case, summary) pairs.")
    split_and_save(pairs)
