from datasets import load_metric
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from typing import List
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import nltk
from nltk.translate import meteor_score
from nltk.tokenize import word_tokenize
from langchain.vectorstores import FAISS
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from peft import PeftModel
from langchain.retrievers import EnsembleRetriever



nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

import pandas as pd
from bert_score import score
from evaluate import load

rouge = load('rouge')
bertscore = load('bertscore')
bleu = load('bleu')

df = pd.read_csv("/kaggle/input/coovid/covid_abstracts.csv")
df = df["abstract"]

embedding_model = SentenceTransformerEmbeddings(model_name='BAAI/bge-large-zh-v1.5')

text_splitter = SemanticChunker(
    embedding_model, breakpoint_threshold_type="percentile"
)
from langchain.docstore.document import Document
docs = [Document(page_content=entry) for entry in df]
docs = text_splitter.split_documents(docs)

len(df)

faiss_db = FAISS.from_documents(docs, embedding_model)

import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
)


model_name = "filipealmeida/Mistral-7B-Instruct-v0.1-sharded"
base_model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.bfloat16,
                                             quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def custom_preprocessing_func(text: str) -> List[str]: # Change the input type to str
    return text.split()

bm25_retriever = BM25Retriever.from_texts(
    [doc.page_content for doc in docs],
    preprocess_func=custom_preprocessing_func
)
bm25_retriever.k = 2

faiss_retriever = faiss_db.as_retriever(
    search_kwargs={'k': 2}
)

peft_model_id = "ProElectro07/Projectxx2"
model = PeftModel.from_pretrained(base_model, peft_model_id)

tokenizer.pad_token = tokenizer.eos_token

##############
text_generation_pipeline = pipeline(
    task="text-generation",
    model=base_model,
    tokenizer=tokenizer,
    temperature=0.3,
    top_k=10,
    top_p=.85,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=500,
    num_return_sequences=1,
    # truncation=False,
    do_sample=True,
    # no_repeat_ngram_size=3,
    # early_stopping=True
)

text_generation_pipeline.model = model

mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[.3, 0.7]
)


ds = pd.read_csv("/kaggle/input/desfcriptionn/validate_dataset.csv")

ds = ds["description"]

predictions = []


prompt_template = ("""[INST]
You are an expert in evaluating the relevance of documents to a query.

Given the query and the 4 documents below:

Query: {question}
Documents: {context}

Task:
1. For each document, label it as "Relevant" if it helps to answer the query, or "Irrelevant" if it does not.

Example:
Relevant
Relevant
Irrelevant
Irrelevant

Please provide the labels only, with each of the 4 document's label on a new line like the above format.

[/INST]
""")



# Create a prompt instance
Prompt_Template = PromptTemplate.from_template(prompt_template)

# Create the RAG chain
RAG_chain = RetrievalQA.from_chain_type(
    mistral_llm,
    retriever=ensemble_retriever,
    chain_type_kwargs={"prompt": Prompt_Template}
)

predictions = []

for query in ds:
    response = RAG_chain({"query":query})
    print(response["result"])
    predictions.append(response["result"])

predictions

p = predictions

responses = p

d = []

for response in responses:
  labels = response.strip().lower().split('\n')
  d.append(labels)

d

for k in d:
  print(len(k))

score = []

test = 0

for i in d:
  t = 0
  if i[0]=="relevant":
    t = t + .4
  if i[1]=="relevant":
    t = t + .3
  if i[2]=="relevant":
    t = t + .2
  if i[3]=="relevant":
    t = t + .1
  score.append(t)
  test = test + t

(score)

d[-5]

(test/50)*100

prompt_template = ("""[INST]
You are an expert in evaluating the relevance of documents to a query.

Given the query and the 4 documents below:

Query: {question}
Documents: {context}

Task:
1. For each document, label it as "Relevant" if it helps to answer the query, or "Irrelevant" if it does not.

Example:
Relevant
Relevant
Irrelevant
Irrelevant

Please provide the labels only, with each of the 4 document's label on a new line like the above format.

[/INST]
""")



# Create a prompt instance
Prompt_Template = PromptTemplate.from_template(prompt_template)

# Create the RAG chain
RAG_chain = RetrievalQA.from_chain_type(
    mistral_llm,
    retriever=faiss_retriever,
    chain_type_kwargs={"prompt": Prompt_Template}
)

predictions = []

for query in ds:
    response = RAG_chain({"query":query})
    print(response["result"])
    predictions.append(response["result"])

predictions

p = predictions

responses = p

d = []

for response in responses:
  labels = response.strip().lower().split('\n')
  d.append(labels)

for k in d:
  print(len(k))

score = []

test = 0

for i in d:
  t = 0
  if i[0]=="relevant":
    t = t + .4
  if i[1]=="relevant":
    t = t + .3
  if i[2]=="relevant":
    t = t + .2
  if i[3]=="relevant":
    t = t + .1
  score.append(t)
  test = test + t

score

(test/50)*100

prompt_template = ("""[INST]
You are an expert in evaluating the relevance of documents to a query.

Given the query and the 4 documents below:

Query: {question}
Documents: {context}

Task:
1. For each document, label it as "Relevant" if it helps to answer the query, or "Irrelevant" if it does not.

Example:
Relevant
Relevant
Irrelevant
Irrelevant

Please provide the labels only, with each of the 4 document's label on a new line like the above format.

[/INST]
""")



# Create a prompt instance
Prompt_Template = PromptTemplate.from_template(prompt_template)

# Create the RAG chain
RAG_chain = RetrievalQA.from_chain_type(
    mistral_llm,
    retriever=bm25_retriever,
    chain_type_kwargs={"prompt": Prompt_Template}
)

predictions = []

for query in ds:
    response = RAG_chain({"query":query})
    print(response["result"])
    predictions.append(response["result"])

p = predictions

responses = p

d = []

for response in responses:
  labels = response.strip().lower().split('\n')
  d.append(labels)

score = []

test = 0

for i in d:
  t = 0
  if i[0]=="relevant":
    t = t + .4
  if i[1]=="relevant":
    t = t + .3
  if i[2]=="relevant":
    t = t + .2
  if i[3]=="relevant":
    t = t + .1
  score.append(t)
  test = test + t

(test/50)*100

