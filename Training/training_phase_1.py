from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import torch
from datasets import load_dataset
from trl import SFTTrainer
from datasets import Dataset


bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = "filipealmeida/Mistral-7B-Instruct-v0.1-sharded"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.bfloat16,
                                             quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

import pandas as pd

dataset_train = pd.read_csv('/content/train_dataset.csv')

import pandas as pd

# Load the dataset
dataset_train = pd.read_csv('train_dataset.csv')

prompt_template = """
<s>[INST] You are a medical expert whose task is to understand a patient's query in Hindi-English code-mixed language and respond like a doctor in English.

Example:
Patient (Hindi-English Code-Mixed): "Hi doctor, mujhe 35 saal ka hoon, koi dil ke problem nahi hai, koi dawai nahi le raha hoon, koi janne wali sujan nahi, abnormal body temperature nahi hai, koi allergies nahi hai kabhi. Raat ko, sirf 3 aur aadhe ghante tak hi sota hoon aur jab uthata hoon tab meri t-shirt pure gili hoti hai jaise main marathon dauda raha tha. Subah aankhein bhi red ho jaati hai. Please doctor, eyes ko dekhiye neeche. Yeh pichle 30 dinon se chal raha hai. Koi idea kya ho sakta hai iska? Bahut bahut shukriya."

English Description: "35-year-old patient reports sleep disturbances, excessive night sweating, and morning eye redness persisting for 30 days. There is red discolouration in the eye."

Doctor's Response: "Based on your description, it seems you're experiencing sleep disturbances, night sweats, and eye redness. These symptoms could be related to various conditions, including sleep disorders, hormonal imbalances, or even certain infections. I recommend scheduling a comprehensive check-up to investigate these symptoms further. We'll need to conduct a physical examination, particularly of your eyes, and possibly some blood tests to rule out any underlying conditions. In the meantime, try to maintain a consistent sleep schedule and ensure your sleeping environment is cool and comfortable. If you notice any worsening of symptoms or develop new ones, please seek immediate medical attention."

Now, please respond to the following patient query:

Patient (Hindi-English Code-Mixed): [{passage}]

English Description: [{description}]

Doctor's Response (in English):
[/INST]
</s>
"""

def formatting_prompts_func(row):
    text = prompt_template.format(passage=row['Codemixed query'], description=row['description'])
    text += row['Doctor'] + '</s>'
    return text

# Apply the formatting function to each row of the DataFrame
dataset_train['formatted_text'] = dataset_train.apply(formatting_prompts_func, axis=1)

# If you need the data in a specific format for your model, you can convert it here
# For example, to get a list of formatted texts:
formatted_texts = dataset_train['formatted_text'].tolist()

tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
        "up_proj", "down_proj", "lm_head"
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)


if isinstance(dataset_train, pd.DataFrame):
    dataset_train = Dataset.from_pandas(dataset_train)

training_arguments = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 5,
        warmup_steps = 3,
        max_steps = 48,
        learning_rate = 5e-5,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.005,
        lr_scheduler_type = "linear",
        output_dir = "outputs",
    )
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_train,
    peft_config=peft_config,
    max_seq_length= 400,
    dataset_text_field="formatted_text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)

trainer.train()
model.push_to_hub("Project", token = "...")

trainer.train()
model.push_to_hub("Projectx2", token = "...")

trainer.train()
model.push_to_hub("Projectx3", token = "...")

trainer.train()
model.push_to_hub("Projectx4", token = "...")
