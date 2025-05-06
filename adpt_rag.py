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

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None



st.title("📄 Legal Document Summarizer (Adaptive RAG)")

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


def generate_response(system_prompt, user_message, model="meta-llama/Llama-3.2-3B-Instruct"):
    return client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
    ).choices[0].message.content


def generate_questions(text_chunk, num_questions=5,
                       model="meta-llama/Llama-3.2-3B-Instruct"):
    system_prompt = (
      "You are an expert at generating relevant questions from text. "
      "Create concise questions that can be answered using only the provided text."
    )
    user_prompt = f"""
    Based on the following text, generate {num_questions} different questions 
    that can be answered using only this text:

    {text_chunk}

    Format your response as a numbered list of questions only.
    """
    resp = client.chat.completions.create(
      model=model,
      temperature=0.7,
      messages=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt}
      ]
    )
    raw = resp.choices[0].message.content.strip()
    questions = []
    for line in raw.split("\n"):
        q = re.sub(r"^\d+\.\s*", "", line).strip()
        if q.endswith("?"):
            questions.append(q)
    return questions

# 2) EMBEDDINGS
def create_embeddings(text, model="BAAI/bge-en-icl"):
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding

def cosine_similarity(a,b):
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

# 3) VECTOR STORE
class SimpleVectorStore:
    def __init__(self):
        self.items = []  # each item is a dict {text, embedding, metadata}

    def add_item(self, text, embedding, metadata):
        self.items.append(dict(text=text, embedding=embedding, metadata=metadata))

    def search(self, query, k=5):
        # existing code: embeds the query itself, returns items without a similarity score
        q_emb = create_embeddings(query)
        scores = [(i, cosine_similarity(q_emb, item["embedding"]))
                  for i,item in enumerate(self.items)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [self.items[i] for i,_ in scores[:k]]

    def similarity_search(self, query_embedding, k=5):
        """
        New method: given a raw embedding vector, return top-k docs
        along with their cosine score in 'similarity'.
        """
        results = []
        for item in self.items:
            sim = float(np.dot(query_embedding, item["embedding"]) /
                        (np.linalg.norm(query_embedding) * np.linalg.norm(item["embedding"])))
            results.append({
                "text":       item["text"],
                "metadata":   item["metadata"],
                "similarity": sim
            })
        # sort by descending similarity and slice
        results.sort(key=lambda doc: doc["similarity"], reverse=True)
        return results[:k]


# 4) DOCUMENT PROCESSOR
def process_document(raw_text,
                     chunk_size=1000,
                     chunk_overlap=200,
                     questions_per_chunk=5):
    # chunk the text
    chunks = []
    for i in range(0, len(raw_text), chunk_size - chunk_overlap):
        chunks.append(raw_text[i:i+chunk_size])
    store = SimpleVectorStore()
    for idx,chunk in enumerate(chunks):
        # chunk embedding
        emb = create_embeddings(chunk)
        store.add_item(chunk, emb, {"type":"chunk","index":idx})
        # generate Qs + their embeddings
        qs = generate_questions(chunk, num_questions=questions_per_chunk)
        for q in qs:
            q_emb = create_embeddings(q)
            store.add_item(q, q_emb, {
              "type":"question",
              "chunk_index":idx,
              "original_chunk": chunk
            })
    return chunks, store

# 5) CONTEXT BUILDER
def prepare_context(results):
    seen = set()
    ctx = []
    # first direct chunks
    for r in results:
        m = r["metadata"]
        if m["type"]=="chunk" and m["index"] not in seen:
            seen.add(m["index"])
            ctx.append(f"Chunk {m['index']}:\n{r['text']}")
    # then referenced by questions
    for r in results:
        m = r["metadata"]
        if m["type"]=="question":
            ci = m["chunk_index"]
            if ci not in seen:
                seen.add(ci)
                ctx.append(f"Chunk {ci} (via Q “{r['text']}”):\n{m['original_chunk']}")
    return "\n\n".join(ctx)

# 6) ANSWER GENERATOR (overrides your old generate_response)
def generate_response_from_context(query, context,
                                   model="meta-llama/Llama-3.2-3B-Instruct"):
    sp = (
      "You are an AI assistant that strictly answers based on the given context. "
      "If the answer cannot be derived directly from the provided context, "
      "respond with: 'I do not have enough information to answer that.'"
    )
    up = f"""
    Context:
    {context}

    Question: {query}

    Please answer the question based only on the context above.
    """
    resp = client.chat.completions.create(
      model=model,
      temperature=0,
      messages=[{"role":"system","content":sp},
                {"role":"user","content":up}]
    )
    return resp.choices[0].message.content



def rag_query_response(prompt, embedding_text):
    chunks = chunk_text_custom(embedding_text)
    chunk_embeddings = create_embeddings(chunks)          # now list of np.arrays
    top_chunks      = semantic_search(prompt, chunks, chunk_embeddings, k=5)
    context_block   = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(top_chunks))

    sys_inst = (
         "You are an AI assistant. Always try to answer from the provided context. "
        "If you aren’t certain, briefly restate the user’s question and point to the most relevant context passages "
        "rather than saying you lack information."
    )
    user_p = f"{context_block}\n\nQuestion: {prompt}"
    return generate_response(sys_inst, user_p)


#############


def classify_query(query, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Classify a query into one of four categories: Factual, Analytical, Opinion, or Contextual.
    
    Args:
        query (str): User query
        model (str): LLM model to use
        
    Returns:
        str: Query category
    """
    # Define the system prompt to guide the AI's classification
    system_prompt = """You are an expert at classifying questions. 
        Classify the given query into exactly one of these categories:
        - Factual: Queries seeking specific, verifiable information.
        - Analytical: Queries requiring comprehensive analysis or explanation.
        - Opinion: Queries about subjective matters or seeking diverse viewpoints.
        - Contextual: Queries that depend on user-specific context.

        Return ONLY the category name, without any explanation or additional text.
    """

    # Create the user prompt with the query to be classified
    user_prompt = f"Classify this query: {query}"
    
    # Generate the classification response from the AI model
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # Extract and strip the category from the response
    category = response.choices[0].message.content.strip()
    
    # Define the list of valid categories
    valid_categories = ["Factual", "Analytical", "Opinion", "Contextual"]
    
    # Ensure the returned category is valid
    for valid in valid_categories:
        if valid in category:
            return valid
    
    # Default to "Factual" if classification fails
    return "Factual"

# def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
#     """
#     Process a document for use with adaptive retrieval.

#     Args:
#     pdf_path (str): Path to the PDF file.
#     chunk_size (int): Size of each chunk in characters.
#     chunk_overlap (int): Overlap between chunks in characters.

#     Returns:
#     Tuple[List[str], SimpleVectorStore]: Document chunks and vector store.
#     """
#     # Extract text from the PDF file
#     print("Extracting text from PDF...")
#     extracted_text = extract_text_from_pdf(pdf_path)
    
#     # Chunk the extracted text
#     print("Chunking text...")
#     chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
#     print(f"Created {len(chunks)} text chunks")
    
#     # Create embeddings for the text chunks
#     print("Creating embeddings for chunks...")
#     chunk_embeddings = create_embeddings(chunks)
    
#     # Initialize the vector store
#     store = SimpleVectorStore()
    
#     # Add each chunk and its embedding to the vector store with metadata
#     for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
#         store.add_item(
#             text=chunk,
#             embedding=embedding,
#             metadata={"index": i, "source": pdf_path}
#         )
    
#     print(f"Added {len(chunks)} chunks to the vector store")
    
#     # Return the chunks and the vector store
#     return chunks, store
	
	
def adaptive_retrieval(query, vector_store, k=4, user_context=None):
    """
    Perform adaptive retrieval by selecting and executing the appropriate strategy.
    
    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store
        k (int): Number of documents to retrieve
        user_context (str): Optional user context for contextual queries
        
    Returns:
        List[Dict]: Retrieved documents
    """
    # Classify the query to determine its type
    query_type = classify_query(query)
    print(f"Query classified as: {query_type}")
    
    # Select and execute the appropriate retrieval strategy based on the query type
    if query_type == "Factual":
        # Use the factual retrieval strategy for precise information
        results = factual_retrieval_strategy(query, vector_store, k)
    elif query_type == "Analytical":
        # Use the analytical retrieval strategy for comprehensive coverage
        results = analytical_retrieval_strategy(query, vector_store, k)
    elif query_type == "Opinion":
        # Use the opinion retrieval strategy for diverse perspectives
        results = opinion_retrieval_strategy(query, vector_store, k)
    elif query_type == "Contextual":
        # Use the contextual retrieval strategy, incorporating user context
        results = contextual_retrieval_strategy(query, vector_store, k, user_context)
    else:
        # Default to factual retrieval strategy if classification fails
        results = factual_retrieval_strategy(query, vector_store, k)
    
    return results  # Return the retrieved documents
	
	
def factual_retrieval_strategy(query, vector_store, k=4):
    """
    Retrieval strategy for factual queries focusing on precision.
    
    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store
        k (int): Number of documents to return
        
    Returns:
        List[Dict]: Retrieved documents
    """
    print(f"Executing Factual retrieval strategy for: '{query}'")
    
    # Use LLM to enhance the query for better precision
    system_prompt = """You are an expert at enhancing search queries.
        Your task is to reformulate the given factual query to make it more precise and 
        specific for information retrieval. Focus on key entities and their relationships.

        Provide ONLY the enhanced query without any explanation.
    """

    user_prompt = f"Enhance this factual query: {query}"
    
    # Generate the enhanced query using the LLM
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # Extract and print the enhanced query
    enhanced_query = response.choices[0].message.content.strip()
    print(f"Enhanced query: {enhanced_query}")
    
    # Create embeddings for the enhanced query
    query_embedding = create_embeddings(enhanced_query)
    
    # Perform initial similarity search to retrieve documents
    initial_results = vector_store.similarity_search(query_embedding, k=k*2)
    
    # Initialize a list to store ranked results
    ranked_results = []
    
    # Score and rank documents by relevance using LLM
    for doc in initial_results:
        relevance_score = score_document_relevance(enhanced_query, doc["text"])
        ranked_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "similarity": doc["similarity"],
            "relevance_score": relevance_score
        })
    
    # Sort the results by relevance score in descending order
    ranked_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # Return the top k results
    return ranked_results[:k]
	
def analytical_retrieval_strategy(query, vector_store, k=4):
    """
    Retrieval strategy for analytical queries focusing on comprehensive coverage.
    
    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store
        k (int): Number of documents to return
        
    Returns:
        List[Dict]: Retrieved documents
    """
    print(f"Executing Analytical retrieval strategy for: '{query}'")
    
    # Define the system prompt to guide the AI in generating sub-questions
    system_prompt = """You are an expert at breaking down complex questions.
    Generate sub-questions that explore different aspects of the main analytical query.
    These sub-questions should cover the breadth of the topic and help retrieve 
    comprehensive information.

    Return a list of exactly 3 sub-questions, one per line.
    """

    # Create the user prompt with the main query
    user_prompt = f"Generate sub-questions for this analytical query: {query}"
    
    # Generate the sub-questions using the LLM
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    
    # Extract and clean the sub-questions
    sub_queries = response.choices[0].message.content.strip().split('\n')
    sub_queries = [q.strip() for q in sub_queries if q.strip()]
    print(f"Generated sub-queries: {sub_queries}")
    
    # Retrieve documents for each sub-query
    all_results = []
    for sub_query in sub_queries:
        # Create embeddings for the sub-query
        sub_query_embedding = create_embeddings(sub_query)
        # Perform similarity search for the sub-query
        results = vector_store.similarity_search(sub_query_embedding, k=2)
        all_results.extend(results)
    
    # Ensure diversity by selecting from different sub-query results
    # Remove duplicates (same text content)
    unique_texts = set()
    diverse_results = []
    
    for result in all_results:
        if result["text"] not in unique_texts:
            unique_texts.add(result["text"])
            diverse_results.append(result)
    
    # If we need more results to reach k, add more from initial results
    if len(diverse_results) < k:
        # Direct retrieval for the main query
        main_query_embedding = create_embeddings(query)
        main_results = vector_store.similarity_search(main_query_embedding, k=k)
        
        for result in main_results:
            if result["text"] not in unique_texts and len(diverse_results) < k:
                unique_texts.add(result["text"])
                diverse_results.append(result)
    
    # Return the top k diverse results
    return diverse_results[:k]
	
	
def opinion_retrieval_strategy(query, vector_store, k=4):
    """
    Retrieval strategy for opinion queries focusing on diverse perspectives.
    
    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store
        k (int): Number of documents to return
        
    Returns:
        List[Dict]: Retrieved documents
    """
    print(f"Executing Opinion retrieval strategy for: '{query}'")
    
    # Define the system prompt to guide the AI in identifying different perspectives
    system_prompt = """You are an expert at identifying different perspectives on a topic.
        For the given query about opinions or viewpoints, identify different perspectives 
        that people might have on this topic.

        Return a list of exactly 3 different viewpoint angles, one per line.
    """

    # Create the user prompt with the main query
    user_prompt = f"Identify different perspectives on: {query}"
    
    # Generate the different perspectives using the LLM
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )
    
    # Extract and clean the viewpoints
    viewpoints = response.choices[0].message.content.strip().split('\n')
    viewpoints = [v.strip() for v in viewpoints if v.strip()]
    print(f"Identified viewpoints: {viewpoints}")
    
    # Retrieve documents representing each viewpoint
    all_results = []
    for viewpoint in viewpoints:
        # Combine the main query with the viewpoint
        combined_query = f"{query} {viewpoint}"
        # Create embeddings for the combined query
        viewpoint_embedding = create_embeddings(combined_query)
        # Perform similarity search for the combined query
        results = vector_store.similarity_search(viewpoint_embedding, k=2)
        
        # Mark results with the viewpoint they represent
        for result in results:
            result["viewpoint"] = viewpoint
        
        # Add the results to the list of all results
        all_results.extend(results)
    
    # Select a diverse range of opinions
    # Ensure we get at least one document from each viewpoint if possible
    selected_results = []
    for viewpoint in viewpoints:
        # Filter documents by viewpoint
        viewpoint_docs = [r for r in all_results if r.get("viewpoint") == viewpoint]
        if viewpoint_docs:
            selected_results.append(viewpoint_docs[0])
    
    # Fill remaining slots with highest similarity docs
    remaining_slots = k - len(selected_results)
    if remaining_slots > 0:
        # Sort remaining docs by similarity
        remaining_docs = [r for r in all_results if r not in selected_results]
        remaining_docs.sort(key=lambda x: x["similarity"], reverse=True)
        selected_results.extend(remaining_docs[:remaining_slots])
    
    # Return the top k results
    return selected_results[:k]
	
def contextual_retrieval_strategy(query, vector_store, k=4, user_context=None):
    """
    Retrieval strategy for contextual queries integrating user context.
    
    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store
        k (int): Number of documents to return
        user_context (str): Additional user context
        
    Returns:
        List[Dict]: Retrieved documents
    """
    print(f"Executing Contextual retrieval strategy for: '{query}'")
    
    # If no user context provided, try to infer it from the query
    if not user_context:
        system_prompt = """You are an expert at understanding implied context in questions.
For the given query, infer what contextual information might be relevant or implied 
but not explicitly stated. Focus on what background would help answering this query.

Return a brief description of the implied context."""

        user_prompt = f"Infer the implied context in this query: {query}"
        
        # Generate the inferred context using the LLM
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        
        # Extract and print the inferred context
        user_context = response.choices[0].message.content.strip()
        print(f"Inferred context: {user_context}")
    
    # Reformulate the query to incorporate context
    system_prompt = """You are an expert at reformulating questions with context.
    Given a query and some contextual information, create a more specific query that 
    incorporates the context to get more relevant information.

    Return ONLY the reformulated query without explanation."""

    user_prompt = f"""
    Query: {query}
    Context: {user_context}

    Reformulate the query to incorporate this context:"""
    
    # Generate the contextualized query using the LLM
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # Extract and print the contextualized query
    contextualized_query = response.choices[0].message.content.strip()
    print(f"Contextualized query: {contextualized_query}")
    
    # Retrieve documents based on the contextualized query
    query_embedding = create_embeddings(contextualized_query)
    initial_results = vector_store.similarity_search(query_embedding, k=k*2)
    
    # Rank documents considering both relevance and user context
    ranked_results = []
    
    for doc in initial_results:
        # Score document relevance considering the context
        context_relevance = score_document_context_relevance(query, user_context, doc["text"])
        ranked_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "similarity": doc["similarity"],
            "context_relevance": context_relevance
        })
    
    # Sort by context relevance and return top k results
    ranked_results.sort(key=lambda x: x["context_relevance"], reverse=True)
    return ranked_results[:k]
	
def score_document_relevance(query, document, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Score document relevance to a query using LLM.
    
    Args:
        query (str): User query
        document (str): Document text
        model (str): LLM model
        
    Returns:
        float: Relevance score from 0-10
    """
    # System prompt to instruct the model on how to rate relevance
    system_prompt = """You are an expert at evaluating document relevance.
        Rate the relevance of a document to a query on a scale from 0 to 10, where:
        0 = Completely irrelevant
        10 = Perfectly addresses the query

        Return ONLY a numerical score between 0 and 10, nothing else.
    """

    # Truncate document if it's too long
    doc_preview = document[:1500] + "..." if len(document) > 1500 else document
    
    # User prompt containing the query and document preview
    user_prompt = f"""
        Query: {query}

        Document: {doc_preview}

        Relevance score (0-10):
    """
    
    # Generate response from the model
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # Extract the score from the model's response
    score_text = response.choices[0].message.content.strip()
    
    # Extract numeric score using regex
    match = re.search(r'(\d+(\.\d+)?)', score_text)
    if match:
        score = float(match.group(1))
        return min(10, max(0, score))  # Ensure score is within 0-10
    else:
        # Default score if extraction fails
        return 5.0
		
def score_document_context_relevance(query, context, document, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Score document relevance considering both query and context.
    
    Args:
        query (str): User query
        context (str): User context
        document (str): Document text
        model (str): LLM model
        
    Returns:
        float: Relevance score from 0-10
    """
    # System prompt to instruct the model on how to rate relevance considering context
    system_prompt = """You are an expert at evaluating document relevance considering context.
        Rate the document on a scale from 0 to 10 based on how well it addresses the query
        when considering the provided context, where:
        0 = Completely irrelevant
        10 = Perfectly addresses the query in the given context

        Return ONLY a numerical score between 0 and 10, nothing else.
    """

    # Truncate document if it's too long
    doc_preview = document[:1500] + "..." if len(document) > 1500 else document
    
    # User prompt containing the query, context, and document preview
    user_prompt = f"""
    Query: {query}
    Context: {context}

    Document: {doc_preview}

    Relevance score considering context (0-10):
    """
    
    # Generate response from the model
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    # Extract the score from the model's response
    score_text = response.choices[0].message.content.strip()
    
    # Extract numeric score using regex
    match = re.search(r'(\d+(\.\d+)?)', score_text)
    if match:
        score = float(match.group(1))
        return min(10, max(0, score))  # Ensure score is within 0-10
    else:
        # Default score if extraction fails
        return 5.0
		
def adaptive_retrieval(query, vector_store, k=4, user_context=None):
    """
    Perform adaptive retrieval by selecting and executing the appropriate strategy.
    
    Args:
        query (str): User query
        vector_store (SimpleVectorStore): Vector store
        k (int): Number of documents to retrieve
        user_context (str): Optional user context for contextual queries
        
    Returns:
        List[Dict]: Retrieved documents
    """
    # Classify the query to determine its type
    query_type = classify_query(query)
    print(f"Query classified as: {query_type}")
    
    # Select and execute the appropriate retrieval strategy based on the query type
    if query_type == "Factual":
        # Use the factual retrieval strategy for precise information
        results = factual_retrieval_strategy(query, vector_store, k)
    elif query_type == "Analytical":
        # Use the analytical retrieval strategy for comprehensive coverage
        results = analytical_retrieval_strategy(query, vector_store, k)
    elif query_type == "Opinion":
        # Use the opinion retrieval strategy for diverse perspectives
        results = opinion_retrieval_strategy(query, vector_store, k)
    elif query_type == "Contextual":
        # Use the contextual retrieval strategy, incorporating user context
        results = contextual_retrieval_strategy(query, vector_store, k, user_context)
    else:
        # Default to factual retrieval strategy if classification fails
        results = factual_retrieval_strategy(query, vector_store, k)
    
    return results  # Return the retrieved 
	
def generate_response(query, results, query_type, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    Generate a response based on query, retrieved documents, and query type.
    
    Args:
        query (str): User query
        results (List[Dict]): Retrieved documents
        query_type (str): Type of query
        model (str): LLM model
        
    Returns:
        str: Generated response
    """
    # Prepare context from retrieved documents by joining their texts with separators
    context = "\n\n---\n\n".join([r["text"] for r in results])
    
    # Create custom system prompt based on query type
    if query_type == "Factual":
        system_prompt = """You are a helpful assistant providing factual information.
    Answer the question based on the provided context. Focus on accuracy and precision.
    If the context doesn't contain the information needed, acknowledge the limitations."""
        
    elif query_type == "Analytical":
        system_prompt = """You are a helpful assistant providing analytical insights.
    Based on the provided context, offer a comprehensive analysis of the topic.
    Cover different aspects and perspectives in your explanation.
    If the context has gaps, acknowledge them while providing the best analysis possible."""
        
    elif query_type == "Opinion":
        system_prompt = """You are a helpful assistant discussing topics with multiple viewpoints.
    Based on the provided context, present different perspectives on the topic.
    Ensure fair representation of diverse opinions without showing bias.
    Acknowledge where the context presents limited viewpoints."""
        
    elif query_type == "Contextual":
        system_prompt = """You are a helpful assistant providing contextually relevant information.
    Answer the question considering both the query and its context.
    Make connections between the query context and the information in the provided documents.
    If the context doesn't fully address the specific situation, acknowledge the limitations."""
        
    else:
        system_prompt = """You are a helpful assistant. Answer the question based on the provided context. If you cannot answer from the context, acknowledge the limitations."""
    
    # Create user prompt by combining the context and the query
    user_prompt = f"""
    Context:
    {context}

    Question: {query}

    Please provide a helpful response based on the context.
    """
    
    # Generate response using the OpenAI client
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )
    
    # Return the generated response content
    return response.choices[0].message.content
	
# Replace your current definition with this:
def rag_with_adaptive_retrieval(
    pdf_file=None,
    document_text=None,
    query="",
    k=4,
    user_context=None,
):
    """
    pdf_file       : a Streamlit UploadedFile, or None if you’re passing raw text
    document_text  : pre-extracted text, or None if you’re uploading a file
    query          : the user’s question or prompt
    """
    # 1️⃣ grab raw_text
    if document_text is not None:
        raw_text = document_text
    elif pdf_file is not None:
        raw_text = extract_text(pdf_file)
    else:
        raise ValueError("Must pass either pdf_file or document_text")

    # 2️⃣ chunk & embed
    chunks, vector_store = process_document(raw_text)

    # 3️⃣ classify & retrieve
    q_type = classify_query(query)
    docs   = adaptive_retrieval(query, vector_store, k=k, user_context=user_context)

    # 4️⃣ generate answer
    answer = generate_response(query, docs, q_type)

    return {
        "query": query,
        "query_type": q_type,
        "retrieved_documents": docs,
        "response": answer
    }


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

    if not st.session_state.processed:
        start_time = time.time()

        raw_text = extract_text(uploaded_file)
        chunks, store = process_document(raw_text)
        st.session_state.vector_store = store
        # …
        st.session_state.processed = True

        summary_dict = hybrid_summary_hierarchical(raw_text)
        embedding_text = prepare_text_for_embedding(summary_dict)
        st.session_state.document_context = embedding_text

        # 2) run adaptive-RAG over the PDF path + role_prompt
        pdf_path = uploaded_file  # Streamlit file‐like can be passed directly
        role_specific_prompt = (
            f"As a {user_role}, summarize the legal document focusing on "
            "the most relevant aspects such as facts, arguments, and judgments "
            "tailored for your role. Include key legal reasoning and timeline of events."
        )
    
        result = rag_with_adaptive_retrieval(
            pdf_file=uploaded_file,
            document_text=raw_text,
            query=role_specific_prompt,
            k=5,
            user_context=None
        )
        initial_summary = result["response"]


        st.session_state.messages.append({"role": "user", "content": f"📤 Uploaded **{uploaded_file.name}**"})
        st.session_state.messages.append({"role": "assistant", "content": initial_summary})

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            display_with_typing_effect(initial_summary)
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
        st.session_state.last_prompt_hash = prompt_hash

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


        # adaptive RAG on the pasted text
        store_text = raw_text  # treat prompt as document
        role_prompt = (
            f"As a {user_role}, summarize the document focusing on facts, "
            "arguments, judgments, plus timeline of events."
        )
       
        result = rag_with_adaptive_retrieval(
        pdf_file=None,
        document_text=raw_text,
        query=role_prompt,
        k=5,
        user_context=None
    )

        initial_summary = result["response"]

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
        # adaptive retrieval on the single‐sentence query

        role_query = f"As a {user_role}, {prompt}"
        store = st.session_state.vector_store
        if st.session_state.vector_store is None:
            st.error("❗ Please upload and process a document first.")
        else:
            store = st.session_state.vector_store
        
        retrieved = adaptive_retrieval(role_query, store, k=5, user_context=None)
        answer   = generate_response(role_query, retrieved, classify_query(role_query))
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


