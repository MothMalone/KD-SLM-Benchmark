import os
import wandb
import transformers
import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from peft import prepare_model_for_kbit_training
import torch
import transformers
torch.cuda.empty_cache()
transformers.logging.set_verbosity_info()
# NOTE: https://docs.wandb.ai/guides/track/#how-it-works
# NOTE: Add the name of the run if needed
wandb.login()
wandb.init(project="KD-SLM")
DATASET_CONTEXT = """
PubMedQA is a dataset and a task that involves Question Answering (QA) using scientific literature from PubMed, which is a free resource that contains millions of articles related to life sciences and biomedical research. PubMedQA specifically focuses on using abstracts and passages from PubMed articles to answer medical and scientific questions.
"""

config = {
    "dataset" : {
        "name": "MothMalone/SLMS-KD-Benchmarks"
    }, 
    "models": {
        "teacher": "meta-llama/Llama-3.2-3B",
        "student": "meta-llama/Llama-3.2-1B"
    },
    "tokenizer": {
    "max_length": 4096,
    "chat_template" : """
            {% for message in messages %}
            {% if loop.first and messages[0]['role'] != 'system' %}
                {{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}
            {% endif %}
            {{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}
            {% endfor %}
            {% if add_generation_prompt %}
            {{ '<|im_start|>assistant\n' }}
            {% endif %}
    """,
    },
    "distillation": {
        "temperature": 2.0,
        "alpha": 0.5
    },
    "training": {
        "report_to": "wandb",
        "output_dir": "./results",
        "num_train_epochs": 10,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "save_steps": 1000,
        "logging_steps": 1,
        "learning_rate": 2e-5,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "resume_from_checkpoint": None,
        "fp16": False,
        "bf16": True
    },
    "model_config": {
        "use_flash_attention": False
    }
}

bits_and_bytes_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# NOTE: take note of sample size of train, val, test
dataset = load_dataset(config['dataset']['name'], 'pubmedqa')
dataset = dataset['train']
student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])
teacher_tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"])

def sharegpt_format(row):
    question =  f"""
    {DATASET_CONTEXT}
    Given the following question from the PubMedQA dataset: 
    question: {row['question']}, 
    with these data:
    context: {row['context']}, 
    long_answer: {row['long_answer']}, 
    Please provide the final decision based on these data and provide the final_decision. It is either
    "yes" or "no" or "maybe".
    """
    return {
        'question' : question
    }
    

# Preprocess data at this
dataset = dataset.map(sharegpt_format)
dataset = dataset.remove_columns(['pubid','context', 'long_answer'])


student_tokenizer.pad_token = student_tokenizer.eos_token
teacher_tokenizer.pad_token = teacher_tokenizer.eos_token



def tokenize_function(row):
    return student_tokenizer(row["question"], truncation=True, max_length=config["tokenizer"]["max_length"], padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=8, remove_columns=["question"])
train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
tokenized_dataset = train_test_split

model_kwargs = {"torch_dtype": torch.bfloat16}
if config["model_config"]["use_flash_attention"]:
    model_kwargs["attn_implementation"] = "flash_attention_2"
    
    
def pad_logits(student_logits, teacher_logits):
    student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)
    if student_size != teacher_size:
        pad_size = abs(student_size - teacher_size)
        pad_tensor = torch.zeros((*teacher_logits.shape[:-1], pad_size), dtype=teacher_logits.dtype, device=teacher_logits.device)
        return (torch.cat([student_logits, pad_tensor], dim=-1), teacher_logits) if student_size < teacher_size else (student_logits, torch.cat([teacher_logits, pad_tensor], dim=-1))
    return student_logits, teacher_logits


student_model = AutoModelForCausalLM.from_pretrained(config["models"]["student"], 
                                            # NOTE: ONLY IF NOT ENOUGH GPU RAM
                                            quantization_config = bits_and_bytes_config,
                                            device_map = "auto")
teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"], 
                                            quantization_config = bits_and_bytes_config,
                                            device_map = "auto")


class LogitsTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        self.teacher_model = self.teacher_model.to(device)
        
        student_model = model.module if hasattr(model, 'module') else model
        teacher_model = self.teacher_model.module if hasattr(self.teacher_model, 'module') else self.teacher_model

        student_outputs = student_model(**inputs)
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)

        custom_loss = self.distillation_loss(model, student_outputs.logits, teacher_outputs.logits, inputs, student_outputs.loss)
        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def distillation_loss(self, model, student_logits, teacher_logits, inputs, original_loss):
        device = next(model.parameters()).device
        student_logits, teacher_logits = pad_logits(student_logits.to(device), teacher_logits.to(device))
        
        student_logits_scaled = student_logits / config["distillation"]["temperature"]
        teacher_logits_scaled = teacher_logits / config["distillation"]["temperature"]

        loss_kd = F.kl_div(
            F.log_softmax(student_logits_scaled, dim=-1),
            F.softmax(teacher_logits_scaled, dim=-1),
            reduction='batchmean'
        ) * (config["distillation"]["temperature"] ** 2) / config["tokenizer"]["max_length"]

        return config["distillation"]["alpha"] * loss_kd + (1 - config["distillation"]["alpha"]) * original_loss



training_arguments = TrainingArguments(**config["training"])

# NOTE: ONLY IF NOT ENOUGH GPU RAM
lora_config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# NOTE: ONLY IF NOT ENOUGH GPU RAM
student_model = prepare_model_for_kbit_training(student_model)
student_model = get_peft_model(student_model, lora_config)

trainer = LogitsTrainer(
    model=student_model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    args=training_arguments,
)

trainer.teacher_model = teacher_model
trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])
trainer.save_model(config["training"]["output_dir"])
