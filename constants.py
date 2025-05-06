from datasets import load_dataset, Dataset
import os

def load_rag_dataset(text_dir, summary_dir):
    records = []
    for fn in os.listdir(text_dir):
        text_path = os.path.join(text_dir, fn)
        sum_path  = os.path.join(summary_dir, fn)
        if os.path.exists(sum_path):
            with open(text_path) as f: txt = f.read().strip()
            with open(sum_path)  as f: summ = f.read().strip()
            records.append({"document": txt, "summary": summ})
    return Dataset.from_list(records)

rag_dataset = load_rag_dataset("case_texts", "case_summaries")


from transformers import RagRetriever, RagTokenizer

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained(
    "facebook/rag-token-nq",
    index_name="custom",
    passages_path="case_texts",      # your corpus
    index_path="case_index.faiss"    # built offline
)
