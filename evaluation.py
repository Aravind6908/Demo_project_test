# evaluate.py
import json
import pandas as pd
import os
from dotenv import load_dotenv

# import exactly the same helper functions you use in Streamlit
from stage_7_smrag2 import (
    extract_text,
    hybrid_summary_hierarchical,
    extract_timeline,
    prepare_text_for_embedding,
    rag_query_response
)

def main():
    # load .env (so your OPENAI_API_KEY etc are available)
    load_dotenv()

    # 1) read ground-truth
    with open("test_query2.json", "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    # 2) map document_id → local file
    doc_paths = {
        "case2": "case2.pdf",
        # add more if you have more documents
    }

    records = []
    for entry in gt_data:
        doc_id = entry["document_id"]
        query  = entry["query"]
        gt_ans = entry["ground_truth_answer"]

        # load & preprocess the document once
        path = doc_paths.get(doc_id)
        if path is None or not os.path.exists(path):
            print(f"⚠️  Missing file for {doc_id}: expected at {path}")
            continue

        with open(path, "rb") as fh:
            raw = extract_text(fh)

        summary_dict = hybrid_summary_hierarchical(raw)
        timeline     = extract_timeline(raw)
        emb_text     = prepare_text_for_embedding(summary_dict, timeline)

        # run the same RAG pipeline you use in Streamlit
        model_ans = rag_query_response(query, emb_text)

        records.append({
            "document_id": doc_id,
            "query": query,
            "ground_truth_answer": gt_ans,
            "model_answer": model_ans
        })
        print(f"✅ Done {doc_id} / “{query}”")

    # 3) push to DataFrame + CSV
    df = pd.DataFrame(records)
    out = "evaluation_results.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\n📝 Saved {len(df)} rows to {out}")

if __name__ == "__main__":
    main()
