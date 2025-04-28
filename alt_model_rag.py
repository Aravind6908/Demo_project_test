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

from openai import OpenAI
import numpy as np


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

if "embedding_text" not in st.session_state:
    st.session_state.embedding_text = None

if "document_context" not in st.session_state:
    st.session_state.document_context = None

if "last_prompt_hash" not in st.session_state:
    st.session_state.last_prompt_hash = None


st.title("📄 Legal Document Summarizer (Alt Model w/o token)")

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

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")
)

# print("API Key:", os.getenv("OPENAI_API_KEY"))  # Temporary for debugging


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


from transformers import pipeline

@st.cache_resource
def load_led_summarizer():
    # Use “allenai/led-base-16384” (or “led-large-16384”)
    return pipeline(
        "summarization",
        model="allenai/led-base-16384",
        tokenizer="allenai/led-base-16384",
        device=0 if torch.cuda.is_available() else -1
    )

led_summarizer = load_led_summarizer()

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

@st.cache_resource
def load_paraphraser():
    # 1) load the slow SentencePiece tokenizer
    tok = AutoTokenizer.from_pretrained(
        "Vamsi/T5_Paraphrase_Paws",
        use_fast=False                # <— force the slow SP tokenizer
    )
    # 2) load the model
    model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    # 3) build a text2text pipeline
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tok,
        device=0 if torch.cuda.is_available() else -1,
        max_length=256,
        num_beams=4,
        do_sample=False
    )

paraphraser = load_paraphraser()


def humanize(text):
    out = paraphraser(f"paraphrase: {text}",
                     max_length=256,
                     num_beams=4,
                     do_sample=False)[0]["generated_text"]
    return out

# then at the end of rag_query_response:




def legalbert_extractive_summary(text, top_ratio=0.2):
    sentences = sent_tokenize(text)
    top_k = max(3, int(len(sentences) * top_ratio))
    if len(sentences) <= top_k:
        return text
    sentence_embeddings = legalbert_model.encode(sentences, convert_to_tensor=True)
    doc_embedding = torch.mean(sentence_embeddings, dim=0)
    cosine_scores = util.pytorch_cos_sim(doc_embedding, sentence_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=top_k)
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
    current_chunk, chunks, summaries = "", [], []
    for sent in sentences:
        if len(tokenizer_led(current_chunk + sent)["input_ids"]) > max_tokens:
            chunks.append(current_chunk)
            current_chunk = sent
        else:
            current_chunk += " " + sent
    if current_chunk:
        chunks.append(current_chunk)
    for chunk in chunks:
        inputs = tokenizer_led(chunk, return_tensors="pt", padding="max_length", truncation=True, max_length=4096)
        global_attention_mask = torch.zeros_like(inputs["input_ids"])
        global_attention_mask[:, 0] = 1
        output = model_led.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            global_attention_mask=global_attention_mask,
            max_length=512,
            min_length=100,
            num_beams=4,
            repetition_penalty=2.0,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=4,
        )
        summaries.append(tokenizer_led.decode(output[0], skip_special_tokens=True))
    return " ".join(summaries)



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


def chunk_text_custom(text, n=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])
    return chunks



def get_embedding(text, model="BAAI/bge-en-icl"):
    """
    From your notebook:
    Creates an embedding for the given text chunk using the BGE-ICL model.
    """
    resp = client.embeddings.create(model=model, input=text)
    return np.array(resp.data[0].embedding)

def create_embeddings(text_chunks, model="BAAI/bge-en-icl"):
    """
    Batch the get_embedding call over your chunks.
    Returns a list of numpy arrays.
    """
    return [get_embedding(chunk, model=model) for chunk in text_chunks]


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))





def semantic_search(query, text_chunks, chunk_embeddings, k=5):
    """
    Compute cosine similarity between the query embedding and each chunk embedding,
    then pick the top-k chunks.
    """
    q_emb = get_embedding(query)
    # simple cosine:
    def cosine(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    scores = [cosine(q_emb, emb) for emb in chunk_embeddings]
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [text_chunks[i] for i in top_idxs]


# def generate_response(system_prompt, user_message, model="meta-llama/Llama-3.2-3B-Instruct"):
#     return client.chat.completions.create(
#         model=model,
#         temperature=0,
#         messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
#     ).choices[0].message.content



# def rag_query_response(prompt, embedding_text):
#     chunks = chunk_text_custom(embedding_text)
#     chunk_embeddings = create_embeddings(chunks)          # now list of np.arrays
#     top_chunks      = semantic_search(prompt, chunks, chunk_embeddings, k=5)
#     context_block   = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(top_chunks))

#     sys_inst = (
#          "You are an AI assistant. Always try to answer from the provided context. "
#         "If you aren’t certain, briefly restate the user’s question and point to the most relevant context passages "
#         "rather than saying you lack information."
#     )
#     user_p = f"{context_block}\n\nQuestion: {prompt}"
#     return generate_response(sys_inst, user_p)

# def rag_query_response(prompt, embedding_text):
#     # 1. retrieve top-k relevant chunks as before
#     chunks           = chunk_text_custom(embedding_text)
#     chunk_embeddings = create_embeddings(chunks)
#     top_chunks       = semantic_search(prompt, chunks, chunk_embeddings, k=5)
#     context_block    = "\n\n".join(f"[{i+1}] {c}" for i,c in enumerate(top_chunks))

#     # 2. build a single input for LED
#     #    we prefix the user’s question so LED knows what to answer
#     led_input = f"{context_block}\n\nQuestion: {prompt}"

#     # 3. call the local LED summarizer
#     #    tune max_length/min_length as you like
#     summary = led_summarizer(
#         led_input,
#         max_length=512,
#         min_length=64,
#         do_sample=False
#     )[0]["summary_text"]

#     return summary

def rag_query_response(prompt, embedding_text):
    # 1) Split & embed your document exactly as before
    chunks           = chunk_text_custom(embedding_text)
    chunk_embeddings = create_embeddings(chunks)

    # 2) Score each chunk against the query & keep top K
    #    Here we also keep the score so we can sort by it
    q_emb  = get_embedding(prompt)
    scores = [float(np.dot(q_emb, e) / (np.linalg.norm(q_emb)*np.linalg.norm(e)))
              for e in chunk_embeddings]
    # get top 5 indices sorted descending
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
    top_chunks = [chunks[i] for i in top_idxs]

    # 3) Build a single LED input, *in order of descending relevance*
    context = "\n\n".join(f"[{i+1}] {c}" for i,c in enumerate(top_chunks))
    led_input = (
        "You are a helpful legal assistant.  "
        "Using ONLY the context below, answer the user’s question "
        "in a clear, conversational tone:\n\n"
        f"{context}\n\n"
        f"Question: {prompt}\n\n"
        "Answer:"
    )

    # 4) Call your local LED summarizer
    raw = led_summarizer(
        led_input,
        max_length=512,
        min_length=64,
        do_sample=False
    )[0]["summary_text"]
    
    summary = humanize(raw)

    return summary


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


# Standard chat input field
prompt = st.chat_input("Type a message...")


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

# Function to prepare text for embedding
# This function combines the extractive and abstractive summaries into a single string for embedding
def prepare_text_for_embedding(summary_dict):
    combined_chunks = []

    for section, content in summary_dict.items():
        ext = content.get("extractive", "").strip()
        abs = content.get("abstractive", "").strip()
        if ext:
            combined_chunks.append(f"{section} - Extractive Summary:\n{ext}")
        if abs:
            combined_chunks.append(f"{section} - Abstractive Summary:\n{abs}")

    return "\n\n".join(combined_chunks)


###################################################################################################################

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



##############################################################################################################

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





#########################################################################################################################


if uploaded_file:
    file_hash = get_file_hash(uploaded_file)
    if file_hash != st.session_state.last_uploaded_hash or reprocess_btn:
        st.session_state.processed = False

    # if is_new_file or reprocess_btn:
    #     st.session_state.processed = False

    if not st.session_state.processed:
        start_time = time.time()
        raw_text = extract_text(uploaded_file)
        summary_dict = hybrid_summary_hierarchical(raw_text)
        # timeline_data = extract_timeline(clean_text(raw_text))
        embedding_text = prepare_text_for_embedding(summary_dict)

        # Generate and display RAG-based summary

        st.session_state.document_context = embedding_text
        
        role_specific_prompt = f"As a {user_role}, summarize the legal document focusing on the most relevant aspects such as facts, arguments, and judgments tailored for your role. Include key legal reasoning and timeline of events where necessary."
        rag_summary = rag_query_response(role_specific_prompt, embedding_text)


        st.session_state.messages.append({"role": "user", "content": f"📤 Uploaded **{uploaded_file.name}**"})
        st.session_state.messages.append({"role": "assistant", "content": rag_summary})

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            display_with_typing_effect(rag_summary)

        processing_time = round((time.time() - start_time) / 60, 2)
        st.info(f"⏱️ Response generated in **{processing_time} minutes**.")

        st.session_state.last_uploaded_hash = file_hash
        st.session_state.processed = True
        st.session_state.last_prompt_hash = None
        save_chat_history(st.session_state.messages)


if prompt:
    words = prompt.split()
    word_count = len(words)
    prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()

    # 1) LONG prompts – echo first, then summarize
    if word_count > 30 and prompt_hash != st.session_state.last_prompt_hash:
        # mark new prompt
        st.session_state.last_prompt_hash = prompt_hash

        # raw_text is just the prompt text
        raw_text = prompt

        st.session_state.messages.append({
            "role": "user",
            "content": f"📥 **Pasted Document Text:**\n\n{limit_text(raw_text, word_limit=500)}"
        })
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(limit_text(raw_text, word_limit=500))

        start_time = time.time()
        summary_dict = hybrid_summary_hierarchical(raw_text)
        emb_text     = prepare_text_for_embedding(summary_dict)

        st.session_state.document_context = emb_text
        st.session_state.processed        = True

        role_prompt = (
            f"As a {user_role}, summarize the document focusing on facts, "
            "arguments, judgments, plus timeline of events."
        )
        initial_summary = rag_query_response(role_prompt, emb_text)

        # 3️⃣ Append & display the assistant’s summary with typing effect
        st.session_state.messages.append({
            "role": "assistant",
            "content": initial_summary
        })
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            display_with_typing_effect(initial_summary)

        st.info(f"⏱️ Summary generated in {round((time.time()-start_time)/60,2)} minutes")
        save_chat_history(st.session_state.messages)

    # 2) SHORT prompts: normal RAG against last context
    elif word_count <= 30 and st.session_state.processed:
        
        role_query = f"As a {user_role}, {prompt}"
        answer     = rag_query_response(role_query, st.session_state.document_context)
        answer = rag_query_response(prompt, st.session_state.document_context)
        st.session_state.messages.append({"role": "user",    "content": prompt})
        st.session_state.messages.append({"role": "assistant","content": answer})
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            display_with_typing_effect(answer)
        save_chat_history(st.session_state.messages)

    # 3) Ingest prompt to start
    else:
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown("❗ Paste at least 30 words of your document to ingest it first.")


######################################################################################################################


# Run this along with streamlit run app.py to evaluate the model's performance on a test set
# Otherwise, comment the below code

# ⇒ EVALUATION HOOK: after the very first summary, fire off evaluate.main() once

# import json
# import pandas as pd
# import threading


# def run_eval(doc_context):

#     with open("test_case1.json", "r", encoding="utf-8") as f:
#             gt_data = json.load(f)

#         # 2) map document_id → local file

#     records = []
#     for entry in gt_data:
#         doc_id = entry["document_id"]
#         query  = entry["query"]
#         gt_ans = entry["ground_truth_answer"]
        
        
#         # model_ans = rag_query_response(query, emb_text)
#         model_ans = rag_query_response(query, doc_context)
        
#         records.append({
#                 "document_id": doc_id,
#                 "query": query,
#                 "ground_truth_answer": gt_ans,
#                 "model_answer": model_ans
#             })
#         print(f"✅ Done {doc_id} / “{query}”")

#         # 3) push to DataFrame + CSV
#         df = pd.DataFrame(records)
#         out = "evaluation_results.csv"
#         df.to_csv(out, index=False, encoding="utf-8")
#         print(f"\n📝 Saved {len(df)} rows to {out}")


# # you could log this somewhere
# def _run_evaluation():
#     try:
#         run_eval()
#     except Exception as e:
#         print("‼️ Evaluation script error:", e)

# if st.session_state.processed and not st.session_state.get("evaluation_launched", False):
#     st.session_state.evaluation_launched = True

#       # inform user
#     st.sidebar.info("🔬 Starting background evaluation run…")

#     # *capture* the context
#     doc_ctx = st.session_state.document_context

#     # spawn the thread, passing doc_ctx in
#     threading.Thread(
#         target=lambda: run_eval(doc_ctx),
#         daemon=True
#     ).start()

#     st.sidebar.success("✅ Evaluation launched — check evaluation_results.csv when done.")

#     # check for file existence & show download button
#     eval_path = os.path.abspath("evaluation_results.csv")
#     if os.path.exists(eval_path):
#         st.sidebar.success(f"✅ Results saved to:\n`{eval_path}`")
#         # load it into a small dataframe (optional)
#         df_eval = pd.read_csv(eval_path)
#         # add a download button
#         st.sidebar.download_button(
#             label="⬇️ Download evaluation_results.csv",
#             data=df_eval.to_csv(index=False).encode("utf-8"),
#             file_name="evaluation_results.csv",
#             mime="text/csv"
#         )
#     else:
#         # if you want, display the cwd so you can inspect it
#         st.sidebar.info(f"Current working dir:\n`{os.getcwd()}`")


