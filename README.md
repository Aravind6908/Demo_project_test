# **Legal Document Summarization**
## **1. Project Overview**

Legal professionals frequently navigate complex, lengthy documents including court rulings, contracts, statutes, and case files. Manually summarizing these documents is not only highly time-consuming—often requiring 4-8 hours for a single case—but also prone to human errors such as missing critical precedents or clauses, potentially leading to costly litigation outcomes. Additionally, smaller law firms and pro bono lawyers frequently lack adequate resources for thorough document analysis, exacerbating legal inequities.

To address these challenges, our project introduces an advanced AI-driven Legal Document Summarizer leveraging state-of-the-art Natural Language Processing (NLP) and Retrieval-Augmented Generation (RAG) technologies. This innovative tool automates the summarization and retrieval of legal documents, significantly enhancing the speed, accuracy, and equity of legal review processes.

### **Key Objectives:**

*   Reduce Time and Costs: Drastically cut down the review time of legal documents by up to 70%, potentially reducing associated paralegal costs by up to 80%.
*   Enhance Accuracy and Precision: Minimize human errors by automatically flagging critical clauses, precedents, and rulings.
*   Improve Accessibility and Equity: Provide equitable access to powerful legal analysis tools, empowering smaller firms and pro bono lawyers.


## **2.Key Functionalities**

### **Document Ingestion & Cleaning**

*   File Support: Handles PDF, DOCX, and TXT files.
*   NLP Processing: Text extraction, tokenization, and cleaning of raw text data to prepare it for advanced NLP tasks.

### **Zero-Shot Sectioning**

*  Utilizes zero-shot classification models (e.g., DistilBART-MNLI) to automatically categorize legal document sections (Facts, Arguments, Judgment).


### **Hybrid Summarization**

#### Extractive Summarization (Legal-BERT)

*   Employs transformer-based Legal-BERT embeddings to identify and extract key sentences that preserve critical information.

#### Abstractive Summarization (LED)

*   Leverages LED-16384 models to generate concise, coherent, and natural-language summaries from extracted content.

### **Embedding & Retrieval (RAG)**

#### Embedding creation (BGE)

*   Generates semantic embeddings using advanced models (e.g., BGE), enhancing semantic relevance in retrieval processes.

#### Question generation (Llama-3.2-3B-Instruct)

*   Uses instruction-tuned generative models (Llama-3.2-3B-Instruct) to create context-aware, precise responses to user queries. 

#### Vector storage and retrieval mechanism

*   Implements efficient vector storage solutions and retrieval techniques powered by BM25 and SBERT for rapid and accurate information retrieval.