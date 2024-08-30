# CM-CliniBOT: A Code‑Mixed Hindi‑English Clinical Chatbot

> **Disclaimer:** CM‑CliniBOT is for informational purposes only and is **not** a substitute for professional medical advice.

CM‑CliniBOT bridges the language gap in medical consultations by understanding and responding to code‑mixed Hindi‑English (Hinglish) queries. Built on a Retrieval‑Augmented Generation (RAG) framework, it retrieves relevant clinical knowledge and generates contextually accurate, empathetic responses.

---

## 📖 Table of Contents

1. [About](#about)  
2. [Project Overview](#project-overview)  
3. [Methodology](#methodology)  
4. [Evaluation](#evaluation)  
5. [Installation](#installation)  

---

## 💡 About

In multilingual societies, patients often describe symptoms in Hinglish. Traditional NLP systems struggle with code‑mixed input, risking miscommunication. CM‑CliniBOT:

- Understands Hinglish queries  
- Retrieves domain‑specific clinical knowledge  
- Generates detailed, empathetic responses  

---

## 🚀 Project Overview

CM‑CliniBOT implements a two‑stage RAG pipeline:

1. **Dataset Creation:** Annotated clinical dialogues reflecting natural Hinglish interactions.  
2. **Model Training:**  
   - Stage 1: Hindi token acquisition using English translations  
   - Stage 2: Response generation from code‑mixed inputs  
3. **Document Retrieval:** Hybrid FAISS + BM25 ensemble on 10,000 COVID‑19 abstracts  
4. **Response Generation:** Translate → Retrieve → Generate  

---

## 🧪 Methodology

### Model Training

| Stage | Input | Output |
|-------|-------|--------|
| 1 | Hinglish query + English translation | Encoded bilingual embeddings |
| 2 | Hinglish query | Clinically appropriate response |

### Document Retrieval

- **Corpus:** 10k COVID‑19 abstracts  
- **Techniques:** FAISS (semantic) + BM25 (token) + multi‑query retriever  

### Response Generation Pipeline

1. Translate Hinglish → English  
2. Retrieve relevant documents  
3. Generate the final answer  

---

## 📊 Evaluation

| Metric | Result | Notes |
|--------|--------|-------|
| BLEU | Moderate | Sensitive to code‑mix nuances |
| ROUGE | Moderate | Similar limitations |
| BERTScore | High | Strong semantic alignment |
| Human Assessment | Preferred over ChatGPT‑4o | More context‑rich & empathetic |

---

## ⚙️ Installation & Usage

```bash
git clone https://github.com/akshatsaxena74/CM-CliniBOT.git
