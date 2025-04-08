import streamlit as st
import shelve
import docx2txt
import PyPDF2
import time  # Used to simulate typing effect
import nltk

import re
import os
import requests
from dotenv import load_dotenv


import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import nltk

nltk.download('punkt')
import hashlib
from nltk import sent_tokenize
nltk.download('punkt_tab')
from transformers import LEDTokenizer, LEDForConditionalGeneration
import torch

st.set_page_config(page_title="Legal Document Summarizer", layout="wide")

st.title("📄 Legal Document Summarizer (zero shot)")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"

# Load chat history
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])

# Save chat history
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

# Function to limit text preview to 500 words
def limit_text(text, word_limit=500):
    words = text.split()
    return " ".join(words[:word_limit]) + ("..." if len(words) > word_limit else "")


# CLEAN AND NORMALIZE TEXT


def clean_text(text):
    # Remove newlines and extra spaces
    text = text.replace('\r\n', ' ').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page number markers like "Page 1 of 10"
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)

    # Remove long dashed or underscored lines
    text = re.sub(r'[_]{5,}', '', text)   # Lines with underscores: _____
    text = re.sub(r'[-]{5,}', '', text)   # Lines with hyphens: -----
    
    # Remove long dotted separators
    text = re.sub(r'[.]{4,}', '', text)   # Dots like "......" or ".............."
    
    # Trim final leading/trailing whitespace
    text = text.strip()

    return text


#######################################################################################################################


# LOADING MODELS FOR DIVIDING TEXT INTO SECTIONS

# Load token from .env file
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")


# def classify_zero_shot_hfapi(text, labels):
#     if not HF_API_TOKEN:
#         return "❌ Hugging Face token not found."

#     headers = {
#         "Authorization": f"Bearer {HF_API_TOKEN}"
#     }

#     payload = {
#         "inputs": text,
#         "parameters": {
#             "candidate_labels": labels
#         }
#     }

#     response = requests.post(
#         "https://api-inference.huggingface.co/models/facebook/bart-large-mnli",
#         headers=headers,
#         json=payload
#     )

#     if response.status_code != 200:
#         return f"❌ Error from HF API: {response.status_code} - {response.text}"

#     result = response.json()
#     return result["labels"][0]  # Return the top label


# # Labels for section classification
# SECTION_LABELS = ["Facts", "Arguments", "Judgment", "Other"]


# def classify_chunk(text):
#     return classify_zero_shot_hfapi(text, SECTION_LABELS)
#     # return result['labels'][0] if result and 'labels' in result else "Other"



# Load once at the top (cache for performance)
@st.cache_resource
def load_local_zero_shot_classifier():
    return pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

local_classifier = load_local_zero_shot_classifier()


SECTION_LABELS = ["Facts", "Arguments", "Judgment", "Other"]

def classify_chunk(text):
    result = local_classifier(text, candidate_labels=SECTION_LABELS)
    return result["labels"][0]


# NEW: NLP-based sectioning using zero-shot classification
def section_by_zero_shot(text):
    sections = {"Facts": "", "Arguments": "", "Judgment": "", "Other": ""}
    sentences = sent_tokenize(text)
    chunk = ""

    for i, sent in enumerate(sentences):
        chunk += sent + " "
        if (i + 1) % 3 == 0 or i == len(sentences) - 1:
            label = classify_chunk(chunk.strip())
            print(f"🔎 Chunk: {chunk[:60]}...\n🔖 Predicted Label: {label}")
            # 👇 Normalize label (title case and fallback)
            label = label.capitalize()
            if label not in sections:
                label = "Other"
            sections[label] += chunk + "\n"
            chunk = ""

    return sections




#######################################################################################################################



# EXTRACTING TEXT FROM UPLOADED FILES

# Function to extract text from uploaded file
def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif file.name.endswith(".docx"):
        full_text = docx2txt.process(file)
    elif file.name.endswith(".txt"):
        full_text = file.read().decode("utf-8")
    else:
        return "Unsupported file type."
    
    return full_text  # Full text is needed for summarization


#######################################################################################################################

# EXTRACTIVE AND ABSTRACTIVE SUMMARIZATION


@st.cache_resource
def load_legalbert():
    return SentenceTransformer("nlpaueb/legal-bert-base-uncased")


legalbert_model = load_legalbert()

# @st.cache_resource
# def load_led():
#     tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
#     model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")
#     return tokenizer, model

# tokenizer_led, model_led = load_led()

@st.cache_resource
def load_fast_bart():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)

bart_summarizer = load_fast_bart()

def legalbert_extractive_summary(text, top_ratio=0.2):
    sentences = sent_tokenize(text)
    top_k = max(3, int(len(sentences) * top_ratio))

    if len(sentences) <= top_k:
        return text

    # Embeddings & scoring
    sentence_embeddings = legalbert_model.encode(sentences, convert_to_tensor=True)
    doc_embedding = torch.mean(sentence_embeddings, dim=0)
    cosine_scores = util.pytorch_cos_sim(doc_embedding, sentence_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=top_k)

    # Preserve original order
    selected_sentences = [sentences[i] for i in sorted(top_results.indices.tolist())]
    return " ".join(selected_sentences)



    # Add LED Abstractive Summarization


# def led_abstractive_summary(text, max_length=512, min_length=100):
#     inputs = tokenizer_led(
#         text, return_tensors="pt", padding="max_length",
#         truncation=True, max_length=4096
#     )
#     global_attention_mask = torch.zeros_like(inputs["input_ids"])
#     global_attention_mask[:, 0] = 1  # Global attention on first token

#     outputs = model_led.generate(
#         inputs["input_ids"],
#         attention_mask=inputs["attention_mask"],
#         global_attention_mask=global_attention_mask,
#         max_length=max_length,
#         min_length=min_length,
#         length_penalty=2.0,
#         num_beams=4
#     )
#     return tokenizer_led.decode(outputs[0], skip_special_tokens=True)



def bart_abstractive_summary_chunked(text, max_chunk_words=700, max_length=256, min_length=60):
    words = text.split()
    summaries = []

    for i in range(0, len(words), max_chunk_words):
        chunk = " ".join(words[i:i+max_chunk_words])
        summary = bart_summarizer(
            chunk, max_length=max_length, min_length=min_length, do_sample=False
        )[0]['summary_text']
        summaries.append(summary)

    return " ".join(summaries)



def hybrid_summary_by_section(text, top_ratio=0.8):
    cleaned_text = clean_text(text)
    sections = section_by_zero_shot(cleaned_text)  # Split into Facts, Arguments, Judgment, Other

    summary_parts = []
    for name, content in sections.items():
        if content.strip():
            # Calculate dynamic number of sentences to extract based on section length
            sentences = sent_tokenize(content)
            top_k = max(3, int(len(sentences) * top_ratio))

            # Extractive summary using Legal-BERT
            extractive = legalbert_extractive_summary(content, 0.8)

            # Abstractive summary using LED (handles long input)
            abstractive = bart_abstractive_summary_chunked(extractive)

            # Combine both
            hybrid = f"📌 **Extractive Summary:**\n{extractive}\n\n🔍 **Abstractive Summary:**\n{abstractive}"
            summary_parts.append(f"### 📘 {name} Section:\n{clean_text(hybrid)}")

    # return "\n\n".join(summary_parts)
    return sections


#######################################################################################################################


# STREAMLIT APP INTERFACE CODE

# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Initialize last_uploaded if not set
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

# Sidebar with a button to delete chat history
with st.sidebar:
    st.subheader("⚙️ Options")
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        st.session_state.last_uploaded = None
        save_chat_history([])

# Display chat messages with a typing effect
def display_with_typing_effect(text, speed=0.005):
    placeholder = st.empty()
    displayed_text = ""
    for char in text:
        displayed_text += char
        placeholder.markdown(displayed_text)
        time.sleep(speed)
    return displayed_text

# Show existing chat messages
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# Standard chat input field
prompt = st.chat_input("Type a message...")

# # Place file uploader AFTER the chat input to keep layout consistent
# uploaded_file = st.file_uploader("📎 Upload a file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

# Place uploader before the chat so it's always visible
with st.container():
    st.subheader("📎 Upload a Legal Document")
    uploaded_file = st.file_uploader("Upload a file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    reprocess_btn = st.button("🔄 Reprocess Last Uploaded File")


# Hashing logic
def get_file_hash(file):
    file.seek(0)
    content = file.read()
    file.seek(0)
    return hashlib.md5(content).hexdigest()



if uploaded_file:
    file_hash = get_file_hash(uploaded_file)
    
    # Check if file is new OR reprocess is triggered
    if file_hash != st.session_state.get("last_uploaded_hash") or reprocess_btn:
        raw_text = extract_text(uploaded_file)
        summary_text = hybrid_summary_by_section(raw_text)

        st.session_state.messages.append({
            "role": "user",
            "content": f"📤 Uploaded **{uploaded_file.name}**"
        })

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            preview_text = f"🧾 **Hybrid Summary of {uploaded_file.name}:**\n\n{summary_text}"
            display_with_typing_effect(clean_text(preview_text), speed=0)

        st.session_state.messages.append({
            "role": "assistant",
            "content": preview_text
        })

        # Save this file hash only if it’s a new upload (avoid overwriting during reprocess)
        if not reprocess_btn:
            st.session_state.last_uploaded_hash = file_hash

        save_chat_history(st.session_state.messages)
        st.rerun()


# Handle chat input and return hybrid summary
if prompt:
    raw_text = prompt
    summary_text = hybrid_summary_by_section(raw_text)
    
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        bot_response = f"📝 **Hybrid Summary of your text:**\n\n{summary_text}"
        display_with_typing_effect(clean_text(bot_response), speed=0)

    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_response
    })

    save_chat_history(st.session_state.messages)
    st.rerun()
