import os
import torch
from datasets import load_dataset
from transformers import (
    LEDTokenizer,
    LEDForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# 1. Paths to JSON files with 'document_text' and 'summary_text'
TRAIN_FILE = "data/train.json"
VAL_FILE   = "data/val.json"
OUTPUT_DIR = "models/led_legal_finetuned"

# 2. Initialize tokenizer and base model
tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
model     = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")

# 3. Preprocessing: add optional few-shot prompt before the summary task
def preprocess_function(examples):
    inputs = [
        "Legal Summarization Prompt:\n" + doc + "\n###\n" + shot
        for doc, shot in zip(
            examples["document_text"],
            examples.get("few_shot_prompt", [""] * len(examples["document_text"]))
        )
    ]
    model_inputs = tokenizer(
        inputs, max_length=4096, truncation=True, padding="max_length"
    )
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["summary_text"], max_length=512, truncation=True, padding="max_length"
        )
    model_inputs["labels"] = labels.input_ids
    return model_inputs

# 4. Load & preprocess datasets
raw = load_dataset(
    "json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE}, field="data"
)
train_ds = raw["train"].map(
    preprocess_function, batched=True,
    remove_columns=raw["train"].column_names
)
val_ds = raw["validation"].map(
    preprocess_function, batched=True,
    remove_columns=raw["validation"].column_names
)

# 5. Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# 6. Training arguments
t_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    logging_steps=100,
    save_total_limit=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),
)

# 7. Trainer (use `processing_class` in place of deprecated `tokenizer` arg)
trainer = Trainer(
    model=model,
    args=t_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
)

# 8. Kick off training
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"✅ Training done! Model saved at {OUTPUT_DIR}")

# 9. Streamlit integration hint:
# Replace your load_led() cache with:
# @st.cache_resource
# def load_led():
#     tok = LEDTokenizer.from_pretrained(OUTPUT_DIR)
#     mdl = LEDForConditionalGeneration.from_pretrained(OUTPUT_DIR)
#     return tok, mdl
