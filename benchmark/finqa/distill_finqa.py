import os
import wandb
import transformers
import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training
import torch
import transformers
import random

torch.cuda.empty_cache()
transformers.logging.set_verbosity_info()

# Set WANDB API key
os.environ["WANDB_API_KEY"] = "1494f76f0db1fdaee413a37d8943d3d1595ebf50"
wandb.login()
wandb.init(
    project="KD-SLM",
    id="finqa-distill-run01",     # Unique run id for finqa
    name="Llama3.2-1B-FinQA-Distill", 
    resume="allow"                   
)

DATASET_CONTEXT = """
FinQA is a financial question answering dataset that requires numerical reasoning over financial documents. Each question is based on financial reports containing tables and text. You need to analyze the provided financial information and answer the question with precise numerical calculations.
"""

config = {
    "dataset" : {
        "name": "MothMalone/SLMS-KD-Benchmarks"
    }, 
    "models": {
        "teacher": "meta-llama/Llama-2-13b-hf",
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
        "output_dir": "./results-finqa-distill",
        "hub_model_id": "MothMalone/Llama3.2-1B-FinQA-Distilled",  
        "push_to_hub": True,
        "hub_strategy": "checkpoint",
        "num_train_epochs": 10,
        "per_device_train_batch_size": 10,
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

# Load FinQA dataset
finqa_ds = load_dataset(config['dataset']['name'], 'finqa')
train_data = finqa_ds['train']
val_data = finqa_ds['validation']

def create_combined_dataset(train_data, val_data, total_samples=1000, seed=42):
    random.seed(seed)
    train_samples = min(500, len(train_data))
    val_samples = min(total_samples - train_samples, len(val_data))
    if val_samples < (total_samples - train_samples):
        remaining_needed = total_samples - train_samples - val_samples
        train_samples = min(train_samples + remaining_needed, len(train_data))
    print(f"Taking {train_samples} samples from train and {val_samples} samples from validation")
    train_indices = random.sample(range(len(train_data)), train_samples)
    val_indices = random.sample(range(len(val_data)), val_samples)
    combined_samples = []
    for idx in train_indices:
        sample = train_data[idx]
        sample_dict = dict(sample)
        sample_dict['source'] = 'train'
        sample_dict['original_index'] = idx
        combined_samples.append(sample_dict)
    for idx in val_indices:
        sample = val_data[idx]
        sample_dict = dict(sample)
        sample_dict['source'] = 'validation'
        sample_dict['original_index'] = idx
        combined_samples.append(sample_dict)
    random.shuffle(combined_samples)
    return combined_samples

# Use the combined dataset for distillation
combined_samples = create_combined_dataset(train_data, val_data, total_samples=1000, seed=42)
import datasets
finqa_combined = datasets.Dataset.from_list(combined_samples)
print(f"Combined FinQA dataset size: {len(finqa_combined)}")

def finqa_format(row):
    pre_text = row.get('pre_text', '')
    table = row.get('table', '')
    post_text = row.get('post_text', '')
    question = row.get('question', '')
    prompt = f"""
    {DATASET_CONTEXT}
    Financial context:
    {pre_text}
    {table}
    {post_text}
    Question: {question}
    Please provide the final answer as a number or text, as appropriate.
    """
    return {'query': prompt}

finqa_combined = finqa_combined.map(finqa_format)
finqa_combined = finqa_combined.remove_columns(['pre_text', 'table', 'post_text', 'question', 'answer', 'program', 'gold_inds', 'program_re', 'id', 'source', 'original_index'])

def tokenize_function(batch):
    enc = tokenizer(batch["query"],
                    truncation=True,
                    padding="max_length",
                    max_length=config["tokenizer"]["max_length"])
    enc["labels"] = enc["input_ids"].copy()
    return enc
tokenized_dataset = finqa_combined.map(tokenize_function, batched=True, remove_columns=["query"])

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
                                            # quantization_config = bits_and_bytes_config, 
                                            device_map = "auto")

teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"], 
                                            # quantization_config = bits_and_bytes_config,  
                                            device_map = "auto")
student_model.resize_token_embeddings(len(tokenizer))
teacher_model.resize_token_embeddings(len(tokenizer))

assert student_model.config.vocab_size == len(tokenizer)
assert tokenizer.pad_token_id < student_model.config.vocab_size
assert teacher_model.config.vocab_size == len(tokenizer)
assert tokenizer.pad_token_id < teacher_model.config.vocab_size

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

trainer = LogitsTrainer(
    model=student_model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    args=training_arguments,
)

trainer.teacher_model = teacher_model
trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])
trainer.save_model(config["training"]["output_dir"])

print("Pushing model to Hugging Face Hub...")
trainer.push_to_hub(commit_message="Knowledge distillation training completed")