import streamlit as st
import shelve
import docx2txt
import PyPDF2
import time  # Used to simulate typing effect
import nltk
import re
import os
import time  # already imported in your code
from dotenv import load_dotenv
import torch
from sentence_transformers import SentenceTransformer, util
nltk.download('punkt')
import hashlib
from nltk import sent_tokenize
nltk.download('punkt_tab')
from transformers import LEDTokenizer, LEDForConditionalGeneration
from transformers import pipeline
import asyncio
import dateutil.parser
from datetime import datetime
import sys
# Fix for RuntimeError: no running event loop on Windows
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


st.set_page_config(page_title="Legal Document Summarizer", layout="wide")



if "processed" not in st.session_state:
    st.session_state.processed = False
if "last_uploaded_hash" not in st.session_state:
    st.session_state.last_uploaded_hash = None
if "chat_prompt_processed" not in st.session_state:
    st.session_state.chat_prompt_processed = False



st.title("📄 Legal Document Summarizer (Embedding Version 1)")

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


# Load once at the top (cache for performance)
@st.cache_resource
def load_local_zero_shot_classifier():
    return pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

local_classifier = load_local_zero_shot_classifier()


SECTION_LABELS = ["Facts", "Arguments", "Judgement", "Others"]

def classify_chunk(text):
    result = local_classifier(text, candidate_labels=SECTION_LABELS)
    return result["labels"][0]


# NEW: NLP-based sectioning using zero-shot classification
def section_by_zero_shot(text):
    sections = {"Facts": "", "Arguments": "", "Judgment": "", "Others": ""}
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
                label = "Others"
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

@st.cache_resource
def load_led():
    tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
    model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")
    return tokenizer, model

tokenizer_led, model_led = load_led()


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


def led_abstractive_summary(text, max_length=512, min_length=100):
    inputs = tokenizer_led(
        text, return_tensors="pt", padding="max_length",
        truncation=True, max_length=4096
    )
    global_attention_mask = torch.zeros_like(inputs["input_ids"])
    global_attention_mask[:, 0] = 1

    outputs = model_led.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        global_attention_mask=global_attention_mask,
        max_length=max_length,
        min_length=min_length,
        num_beams=4,                      # Use beam search
        repetition_penalty=2.0,           # Penalize repetition
        length_penalty=1.0,
        early_stopping=True,
        no_repeat_ngram_size=4            # Prevent repeated phrases
    )

    return tokenizer_led.decode(outputs[0], skip_special_tokens=True)



def led_abstractive_summary_chunked(text, max_tokens=3000):
    sentences = sent_tokenize(text)
    current_chunk = ""
    chunks = []
    for sent in sentences:
        if len(tokenizer_led(current_chunk + sent)["input_ids"]) > max_tokens:
            chunks.append(current_chunk)
            current_chunk = sent
        else:
            current_chunk += " " + sent
    if current_chunk:
        chunks.append(current_chunk)

    summaries = []
    for chunk in chunks:
        summaries.append(led_abstractive_summary(chunk))  # Call your LED summary function here

    return " ".join(summaries)



# EXTRACT TIMELINE OF EVENTS


def extract_timeline(text):
    sentences = sent_tokenize(text)
    timeline = []

    for sentence in sentences:
        try:
            # Try fuzzy parsing on the sentence
            parsed = dateutil.parser.parse(sentence, fuzzy=True)

            # Validate year: exclude years before 1950 unless explicitly whitelisted
            current_year = datetime.now().year
            if 1900 <= parsed.year <= current_year + 5:
                # Additional filtering: discard misleading past years unless contextually valid
                if parsed.year < 1950 and parsed.year not in [2020, 2022, 2023]:
                    continue

                # Further validation: ignore obviously wrong patterns like years starting with 0
                if re.match(r"^0\d{3}$", str(parsed.year)):
                    continue

                # Passed all checks
                timeline.append((parsed.date(), sentence.strip()))
        except Exception:
            continue

    # Remove duplicates and sort
    unique_timeline = list(set(timeline))
    return sorted(unique_timeline, key=lambda x: x[0])



def format_timeline_for_chat(timeline_data):
    if not timeline_data:
        return "_No significant timeline events detected._"
    
    formatted = "🗓️ **Timeline of Events**\n\n"
    for date, event in timeline_data:
        formatted += f"**{date.strftime('%Y-%m-%d')}**: {event}\n\n"
    return formatted.strip()



def hybrid_summary_hierarchical(text, top_ratio=0.8):
    cleaned_text = clean_text(text)
    sections = section_by_zero_shot(cleaned_text)

    structured_summary = {}  # <-- hierarchical summary here

    for name, content in sections.items():
        if content.strip():
            # Extractive summary
            extractive = legalbert_extractive_summary(content, top_ratio)

            # Abstractive summary
            abstractive = led_abstractive_summary_chunked(extractive)

            # Store in dictionary (hierarchical structure)
            structured_summary[name] = {
                "extractive": extractive,
                "abstractive": abstractive
            }

    return structured_summary


#######################################################################################################################


# STREAMLIT APP INTERFACE CODE

# Initialize or load chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# Initialize last_uploaded if not set
if "last_uploaded" not in st.session_state:
    st.session_state.last_uploaded = None

with st.sidebar:
    st.subheader("⚙️ Options")
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        st.session_state.last_uploaded = None
        st.session_state.processed = False
        st.session_state.chat_prompt_processed = False
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



# Hashing logic
def get_file_hash(file):
    file.seek(0)
    content = file.read()
    file.seek(0)
    return hashlib.md5(content).hexdigest()


########################################################################################################################

user_role = st.sidebar.selectbox(
    "🎭 Select Your Role for Custom Summary",
    ["General", "Judge", "Lawyer", "Student"]
)


def role_based_filter(section, summary, role):
    if role == "General":
        return summary
    
    filtered_summary = {
        "extractive": "",
        "abstractive": ""
    }

    if role == "Judge" and section in ["Judgement", "Facts"]:
        filtered_summary = summary
    elif role == "Lawyer" and section in ["Arguments", "Facts"]:
        filtered_summary = summary
    elif role == "Student" and section in ["Facts"]:
        filtered_summary = summary

    return filtered_summary



# Process uploaded file

# if uploaded_file:
#     file_hash = get_file_hash(uploaded_file)
#     is_new_file = file_hash != st.session_state.last_uploaded_hash

#     if is_new_file or reprocess_btn:
#         st.session_state.processed = False

#     if not st.session_state.processed:
#         start_time = time.time()
#         raw_text = extract_text(uploaded_file)
#         summary_dict = hybrid_summary_hierarchical(raw_text)

#         st.session_state.messages.append({
#             "role": "user",
#             "content": f"📤 Uploaded **{uploaded_file.name}**"
#         })

#         preview_text = f"🧾 **Hybrid Summary of {uploaded_file.name}:**\n\n"
#         for section in ["Facts", "Arguments", "Judgement", "Others"]:
#             if section in summary_dict:
#                 filtered = role_based_filter(section, summary_dict[section], user_role)
#                 extractive = filtered.get("extractive", "").strip()
#                 abstractive = filtered.get("abstractive", "").strip()

#                 if not extractive and not abstractive:
#                     continue

#                 preview_text += f"### 📘 {section} Section\n"
#                 preview_text += f"📌 **Extractive Summary:**\n{extractive if extractive else '_No content extracted._'}\n\n"
#                 preview_text += f"🔍 **Abstractive Summary:**\n{abstractive if abstractive else '_No summary generated._'}\n\n"

#         with st.chat_message("assistant", avatar=BOT_AVATAR):
#             display_with_typing_effect(clean_text(preview_text), speed=0)

#         # ⏱️ Timeline & Response Time
#         timeline_data = extract_timeline(clean_text(raw_text))
#         if timeline_data:
#             st.subheader("🗓️ Timeline of Events")
#             for date, event in timeline_data:
#                 with st.expander(f"{date.strftime('%Y-%m-%d')}"):
#                     st.write(f"**Event:** {event}")
#         else:
#             st.info("No significant timeline events detected.")

#         # Save timeline in chat history
#         timeline_text = format_timeline_for_chat(timeline_data)
#         st.session_state.messages.append({
#             "role": "assistant",
#             "content": timeline_text
#         })


#         processing_time = round((time.time() - start_time)/60, 2)
#         st.info(f"⏱️ Response generated in **{processing_time} minutes**.")

#         st.session_state.messages.append({
#             "role": "assistant",
#             "content": clean_text(preview_text)
#         })

#         save_chat_history(st.session_state.messages)
#         st.session_state.last_uploaded_hash = file_hash
#         st.session_state.processed = True

#

# if prompt and not st.session_state.chat_prompt_processed:
#     start_time = time.time()
#     raw_text = prompt

#     summary_dict = hybrid_summary_hierarchical(raw_text)

#     st.session_state.messages.append({
#         "role": "user",
#         "content": prompt
#     })

#     preview_text = f"🧾 **Hybrid Summary of User Input:**\n\n"
#     for section in ["Facts", "Arguments", "Judgement", "Others"]:
#         if section in summary_dict:
#             filtered = role_based_filter(section, summary_dict[section], user_role)
#             extractive = filtered.get("extractive", "").strip()
#             abstractive = filtered.get("abstractive", "").strip()

#             if not extractive and not abstractive:
#                 continue

#             preview_text += f"### 📘 {section} Section\n"
#             preview_text += f"📌 **Extractive Summary:**\n{extractive if extractive else '_No content extracted._'}\n\n"
#             preview_text += f"🔍 **Abstractive Summary:**\n{abstractive if abstractive else '_No summary generated._'}\n\n"

#     with st.chat_message("assistant", avatar=BOT_AVATAR):
#         display_with_typing_effect(clean_text(preview_text), speed=0)

#     timeline_data = extract_timeline(clean_text(raw_text))
#     if timeline_data:
#         st.subheader("🗓️ Timeline of Events")
#         for date, event in timeline_data:
#             with st.expander(f"{date.strftime('%Y-%m-%d')}"):
#                 st.write(f"**Event:** {event}")
#     else:
#         st.info("No significant timeline events detected.")

#     # Save timeline in chat history
#     timeline_text = format_timeline_for_chat(timeline_data)
#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": timeline_text
#     })


#     processing_time = round((time.time() - start_time)/60, 2)
#     st.info(f"⏱️ Response generated in **{processing_time} minutes**.")

#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": clean_text(preview_text)
#     })

#     save_chat_history(st.session_state.messages)
#     st.session_state.chat_prompt_processed = True


# Store cleaned text and FAISS index only when document is processed

# Embedding for chunking



def chunk_text(text, max_tokens=100):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

import faiss
import numpy as np


def build_faiss_index(chunks):
    embedder = load_embedder()
    embeddings = embedder.encode(chunks, convert_to_tensor=False)
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    st.session_state["embedder"] = embedder
    return index, chunks  # ✅ Return both


def retrieve_top_k(query, chunks, index, k=3):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), k)
    return [chunks[i] for i in I[0]]


def summarize_query_response(query, chunks, index):
    top_chunks = retrieve_top_k(query, chunks, index)
    combined_text = "\n".join(top_chunks)
    return led_abstractive_summary_chunked(combined_text)

# Document upload section
with st.container():
    st.subheader("📎 Upload a Legal Document")
    uploaded_file = st.file_uploader("Upload a file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    reprocess_btn = st.button("🔄 Reprocess Last Uploaded File")

if uploaded_file:
    file_hash = hashlib.md5(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    is_new_file = file_hash != st.session_state.last_uploaded_hash
    if is_new_file or reprocess_btn:
        st.session_state.processed = False
    if not st.session_state.processed:
        raw_text = extract_text(uploaded_file)
        cleaned_text = clean_text(raw_text)
        chunks = chunk_text(cleaned_text)
        index, chunked_data = build_faiss_index(chunks)
        st.session_state["faiss_chunks"] = chunked_data
        st.session_state["faiss_index"] = index
        st.session_state["last_uploaded_hash"] = file_hash
        st.session_state.processed = True
        st.session_state.messages.append({"role": "user", "content": f"📤 Uploaded **{uploaded_file.name}**"})

# Display chat history
for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

prompt = st.chat_input("Type a message...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    if "faiss_chunks" in st.session_state and "faiss_index" in st.session_state:
        chunks = st.session_state["faiss_chunks"]
        index = st.session_state["faiss_index"]
        answer = summarize_query_response(prompt, chunks, index)
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            display_with_typing_effect(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        save_chat_history(st.session_state.messages)
    else:
        st.warning("Please upload and process a document before asking questions.")


########################################################################################################################