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



def calculate_bleurt(predictions, references):
    bleurt = load_metric("bleurt", config_name="BLEURT-20")
    results = bleurt.compute(predictions=predictions, references=references)
    return results['scores']


def calculate_meteor(prediction, reference):
    # Tokenize the prediction and reference
    prediction_tokens = word_tokenize(prediction)
    reference_tokens = word_tokenize(reference)
    return meteor_score.meteor_score([reference_tokens], prediction_tokens)

def evaluate_predictions(predictions, references):
    bleurt_scores = calculate_bleurt(predictions, references)
    meteor_scores = [calculate_meteor(pred, ref) for pred, ref in zip(predictions, references)]

    avg_bleurt = sum(bleurt_scores) / len(bleurt_scores)
    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    return {
        "BLEURT": avg_bleurt,
        "METEOR": avg_meteor,
        "BLEURT_scores": bleurt_scores,
        "METEOR_scores": meteor_scores
    }


import pandas as pd
df = pd.read_csv("/kaggle/input/coovid/covid_abstracts.csv")
df = df["abstract"]

embedding_model = SentenceTransformerEmbeddings(model_name='BAAI/bge-large-zh-v1.5')

text_splitter = SemanticChunker(
    embedding_model, breakpoint_threshold_type="percentile"
)
from langchain.docstore.document import Document
docs = [Document(page_content=entry) for entry in df]
docs = text_splitter.split_documents(docs)


faiss_db = FAISS.from_documents(docs, embedding_model)

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

test = pd.read_csv("/kaggle/input/testtt/testt.csv")

peft_model_id = "ProElectro07/Projectxx2"
model = PeftModel.from_pretrained(base_model, peft_model_id)

tokenizer.pad_token = tokenizer.eos_token

from langchain.chains import SimpleSequentialChain, LLMChain

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

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=ensemble_retriever, llm=mistral_llm
)

references = []
for query in test["response"]:
  references.append(query)

from bert_score import score
from evaluate import load
rouge = load('rouge')
bertscore = load('bertscore')
bleu = load('bleu')

###############3
translation_prompt = PromptTemplate(
    template="""
<s>[INST]
You are an experienced linguistic expert and an assitant to a doctor who only understand english.
You will be given code-mixed hindi-english queries and you need to translate them into english.

Here is an example of a good response:
Patient: Doctor sahib, mumjhe bukhar aur khansi ho rahi hai, kya yeh covid ka lakshan ho sakta hai?
Reponse: Doctor, I have fever and cough, are these things symptoms of covid?

Now, please translate the patient's query below in a similar manner, in English:
Patient's question: [{input}]
Provide an accurate translated query which could :
[/INST] </s>
"""
)

translation_chain = LLMChain(llm=mistral_llm, prompt=translation_prompt)

#########################
prompt_template = """<s>[INST]
You are an experienced COVID-19 doctor, who will help out the patient with their queries.

Here is an example of a good response:
Patient: Doctor, I have fever and cough, are these things symptoms of covid?
Doctor: Yes, fever and cough can indeed be symptoms of COVID-19. I recommend you get tested for COVID-19 as soon as possible. In the meantime, please isolate yourself at home, rest, and drink plenty of fluids. Monitor your symptoms closely, and if they worsen, especially if you have difficulty breathing, seek medical attention immediately.

Patient's question: {question}

Additional context that might help the patient's query: {context}

Provide a clear and concise response to the patient's:
[/INST] </s>
"""

# Create a prompt instance
Prompt_Template = PromptTemplate.from_template(prompt_template)

# Create the RAG chain
RAG_chain = RetrievalQA.from_chain_type(
    mistral_llm,
    retriever=retriever_from_llm,
    chain_type_kwargs={"prompt": Prompt_Template}
)
combined_chain = SimpleSequentialChain(
    chains=[translation_chain, RAG_chain],
    verbose=True
)

predictions = []
count = 0
for query in test["queries"]:
    response = combined_chain({"input": query})
    # count = count + 1
    # response = RAG_chain({"query": query["text"]})
    predictions.append(response["output"])
    print(response["output"])

predictions

predictions_fixed = [p for p in predictions]
references_fixed = [r for r in references]

rouge_scores = rouge.compute(predictions=predictions_fixed, references=references_fixed)
bleu_score = bleu.compute(predictions=predictions_fixed, references=references_fixed)

P, R, F1 = score(predictions_fixed, references_fixed, lang="en", verbose=True)

bert_scores = {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}

print("ROUGE scores:", rouge_scores)
print("BERT scores:", bert_scores)
print("BLEU score:", bleu_score)

from langchain.retrievers.multi_query import MultiQueryRetriever

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[1, 0]
)

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=ensemble_retriever, llm=mistral_llm
)

# Create the RAG chain
RAG_chain = RetrievalQA.from_chain_type(
    mistral_llm,
    retriever=retriever_from_llm,
    chain_type_kwargs={"prompt": Prompt_Template}
)
combined_chain = SimpleSequentialChain(
    chains=[translation_chain, RAG_chain],
    verbose=True
)

predictions = []
count = 0
for query in test["queries"]:
    response = combined_chain({"input": query})
    predictions.append(response["output"])
    print(response["output"])

predictions_fixed = [p for p in predictions]
references_fixed = [r for r in references]

rouge_scores = rouge.compute(predictions=predictions_fixed, references=references_fixed)
bleu_score = bleu.compute(predictions=predictions_fixed, references=references_fixed)

P, R, F1 = score(predictions_fixed, references_fixed, lang="en", verbose=True)

bert_scores = {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}

#########################
prompt_template = """<s>[INST]
You are an experienced COVID-19 doctor, who will help out the patient with their queries.

Here is an example of a good response:
Patient: Doctor, I have fever and cough, are these things symptoms of covid?
Doctor: Yes, fever and cough can indeed be symptoms of COVID-19. I recommend you get tested for COVID-19 as soon as possible. In the meantime, please isolate yourself at home, rest, and drink plenty of fluids. Monitor your symptoms closely, and if they worsen, especially if you have difficulty breathing, seek medical attention immediately.

Patient's question: {question}

Additional context that might help the patient's query: {context}

Provide a clear and concise response to the patient's:
[/INST] </s>
"""

# Create a prompt instance
Prompt_Template = PromptTemplate.from_template(prompt_template)

# Create the RAG chain
check = RetrievalQA.from_chain_type(
    mistral_llm,
    retriever=retriever_from_llm,
    chain_type_kwargs={"prompt": Prompt_Template}
)
combined_chain = SimpleSequentialChain(
    chains=[translation_chain, RAG_chain],
    verbose=True
)

predictions = []
count = 0
for query in test["queries"]:
    response = combined_chain({"input": query})
    # count = count + 1
    # response = RAG_chain({"query": query["text"]})
    predictions.append(response["output"])
    print(response["output"])