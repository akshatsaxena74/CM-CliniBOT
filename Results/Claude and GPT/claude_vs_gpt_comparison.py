from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
device = 'cuda' if torch.cuda.is_available() else 'cpu'

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from peft import PeftModel
from langchain.retrievers import EnsembleRetriever

import pandas as pd
test1 = pd.read_csv("/content/claude.csv")
test2 = pd.read_csv("/content/gptt.csv")

test1.columns

predictions1 = []
for p in test1["Responses"]:
  predictions1.append(p)

predictions2 = []
for p in test2["responses"]:
  predictions2.append(p)

references = []

df = pd.read_csv("/content/testt.csv")

for r in df["response"]:
  references.append(r)

references = references[:50]

from bert_score import score
from evaluate import load
rouge = load('rouge')
bertscore = load('bertscore')
bleu = load('bleu')

predictions_fixed = [p for p in predictions1]
references_fixed = [r for r in references]

rouge_scores = rouge.compute(predictions=predictions_fixed, references=references_fixed)
bleu_score = bleu.compute(predictions=predictions_fixed, references=references_fixed)

P, R, F1 = score(predictions_fixed, references_fixed, lang="en", verbose=True)

bert_scores = {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}

print("ROUGE scores:", rouge_scores)
print("BERT scores:", bert_scores)
print("BLEU score:", bleu_score)

predictions_fixed = [p for p in predictions2]
references_fixed = [r for r in references]

rouge_scores = rouge.compute(predictions=predictions_fixed, references=references_fixed)
bleu_score = bleu.compute(predictions=predictions_fixed, references=references_fixed)

P, R, F1 = score(predictions_fixed, references_fixed, lang="en", verbose=True)

bert_scores = {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}

print("ROUGE scores:", rouge_scores)
print("BERT scores:", bert_scores)
print("BLEU score:", bleu_score)

