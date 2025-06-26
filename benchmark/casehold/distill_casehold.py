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

# Set WANDB API key
os.environ["WANDB_API_KEY"] = "1494f76f0db1fdaee413a37d8943d3d1595ebf50"
# NOTE: https://docs.wandb.ai/guides/track/#how-it-works
wandb.login()
wandb.init(
    project="KD-SLM",
    id="casehold-distill-L3-family",     
    name="Llama3.2-3B-to-1B-Casehold-Distill", 
    resume="allow"                   
)

DATASET_CONTEXT = """
CaseHOLD is a dataset for legal reasoning that involves multiple-choice questions based on legal cases. The task is to identify the correct legal holding (the key legal principle or rule) from a set of candidate holdings, given a citing prompt that describes a legal scenario or case context.
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
        "output_dir": "./results-casehold-distill-new",
        "hub_model_id": "MothMalone/Llama3.2-3B-to-1B-Casehold-Distilled",  
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

# Load dataset and check sample sizes
dataset = load_dataset(config['dataset']['name'], 'casehold')
dataset = dataset['train']
print(f"Total dataset size: {len(dataset)}")
print(f"Using full dataset for training")
# NOTE: If you want to limit sample size for testing, uncomment below:
dataset = dataset.select(range(1000))  # Use only first 10k samples
print(f"Limited dataset size: {len(dataset)}")
tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"])
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

def sharegpt_format(row):
    question = f"""
    {DATASET_CONTEXT}
    Given the following legal case scenario from the CaseHOLD dataset:
    citing_prompt: {row['citing_prompt']},
    with these holding options:
    0: {row['holding_0']}
    1: {row['holding_1']}
    2: {row['holding_2']}
    3: {row['holding_3']}
    4: {row['holding_4']}
    Please provide the final decision based on the legal scenario and determine which holding is correct. 
    The answer should be the index number (0, 1, 2, 3, or 4) of the correct holding.
    """
    return {
        'question': question
    }

# Preprocess data at this
dataset = dataset.map(sharegpt_format)
dataset = dataset.remove_columns(['citing_prompt', 'holding_0', 'holding_1', 'holding_2', 'holding_3', 'holding_4', 'label'])



def tokenize_function(batch):
    enc = tokenizer(batch["question"],
                    truncation=True,
                    padding="max_length",
                    max_length=config["tokenizer"]["max_length"])
    enc["labels"] = enc["input_ids"].copy()
    return enc
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["question"])


train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
tokenized_dataset = train_test_split

model_kwargs = {"torch_dtype": torch.bfloat16}
if config["model_config"]["use_flash_attention"]:
    model_kwargs["attn_implementation"] = "flash_attention_2"
    
# def pad_logits(student_logits, teacher_logits):
#     student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)
#     # if student_size != teacher_size:
#     #     pad_size = abs(student_size - teacher_size)
#     #     pad_tensor = torch.zeros((*teacher_logits.shape[:-1], pad_size), dtype=teacher_logits.dtype, device=teacher_logits.device)
#     #     return (torch.cat([student_logits, pad_tensor], dim=-1), teacher_logits) if student_size < teacher_size else (student_logits, torch.cat([teacher_logits, pad_tensor], dim=-1))
#     return student_logits, teacher_logits

# NOTE: Remove quantization_config if you have enough GPU RAM (>40GB) for better performance
student_model = AutoModelForCausalLM.from_pretrained(config["models"]["student"], 
                                            # quantization_config = bits_and_bytes_config, 
                                            device_map = "auto")

teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"], 
                                            # quantization_config = bits_and_bytes_config,  
                                            device_map = "auto")
# student_model.resize_token_embeddings(len(tokenizer))
# teacher_model.resize_token_embeddings(len(tokenizer))


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
        # student_logits, teacher_logits = pad_logits(student_logits.to(device), teacher_logits.to(device))
        
        student_logits_scaled = student_logits / config["distillation"]["temperature"]
        teacher_logits_scaled = teacher_logits / config["distillation"]["temperature"]

        loss_kd = F.kl_div(
            F.log_softmax(student_logits_scaled, dim=-1),
            F.softmax(teacher_logits_scaled, dim=-1),
            reduction='batchmean'
        ) * (config["distillation"]["temperature"] ** 2) / config["tokenizer"]["max_length"]

        return config["distillation"]["alpha"] * loss_kd + (1 - config["distillation"]["alpha"]) * original_loss

training_arguments = TrainingArguments(**config["training"])


# NOTE: Remove LoRA (this entire section) if you have enough GPU RAM (>40GB) for full fine-tuning
# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=8,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM"
# )

# Prepare model for quantized training and apply LoRA
# NOTE: Skip these two lines if you removed quantization above
# student_model = prepare_model_for_kbit_training(student_model)
# student_model = get_peft_model(student_model, lora_config)

trainer = LogitsTrainer(
    model=student_model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    args=training_arguments,
)

inputs = next(iter(trainer.get_train_dataloader()))
max_id = inputs["input_ids"].max().item()
vocab_size = student_model.config.vocab_size   # you should check BOTH student & teacher
print(f"Max token ID in batch: {max_id}, vocab size: {vocab_size}")
assert max_id < vocab_size, "Found out‐of‐range token ID!"

trainer.teacher_model = teacher_model
trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])
trainer.save_model(config["training"]["output_dir"])

# Push the final model to Hugging Face Hub
print("Pushing model to Hugging Face Hub...")
trainer.push_to_hub(commit_message="Knowledge distillation training completed")