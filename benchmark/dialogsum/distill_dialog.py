import os
import torch
import torch.nn.functional as F
import csv

from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl
import yaml
from datasets import load_dataset, concatenate_datasets



import csv
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

class SimpleCSVLoggerCallback(TrainerCallback):
    def __init__(self, csv_path="training_log.csv"):
        self.csv_path = csv_path
        self.header_written = False

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return
        logs = {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}
        if not logs:
            return
        write_header = not self.header_written and not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=logs.keys())
            if write_header:
                writer.writeheader()
                self.header_written = True
            writer.writerow(logs)

# Configuration
config = {
    "project_name": "llama32-distillation-dialogsum",
    "dataset": {
        "name": "knkarthick/dialogsum",
        "split": "train",
        # "num_samples": 1000,  # You can pass a number here to limit the number of samples to use.
        "seed": 42
    },
    "models": {
        "teacher": "meta-llama/Llama-3.2-3B-Instruct",
        "student": "meta-llama/Llama-3.2-1B-Instruct"
    },
    "tokenizer": {
        "max_length": 2048,  # Reduced for DialogSum which has shorter conversations
        "chat_template": None  # Will use the default Llama chat template
    },
    "training": {
        "output_dir": "./results/llama32-distilled",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "save_steps": 500,
        "logging_steps": 50,
        "learning_rate": 1e-5,  # Lower learning rate for Llama
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "resume_from_checkpoint": None,  # Set to a path or True to resume from the latest checkpoint
        "fp16": False,
        "bf16": True,
        "dataloader_num_workers": 4,
        "remove_unused_columns": False,
        "eval_strategy": "steps",
        "eval_steps": 500,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False
    },
    "distillation": {
        "temperature": 3.0,  
        "alpha": 0.7  
    },
    "model_config": {
        "use_flash_attention": False
    }
    # "spectrum": {
    #     "layers_to_unfreeze": "/workspace/spectrum/snr_results_Llama-3.2-1B_unfrozenparameters_50percent.yaml" # You can pass a spectrum yaml file here to freeze layers identified by spectrum.
    # }
}


# Load and preprocess dataset
print("Loading DialogSum dataset...")
full = load_dataset(config["dataset"]["name"])
train = full["train"].shuffle(seed=config["dataset"]["seed"]).select(range(1000))
validation = full["validation"]

train_dataset = train
val_dataset = validation

# Load tokenizers
teacher_tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"])
student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])

# Set padding tokens if not present
if teacher_tokenizer.pad_token is None:
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
if student_tokenizer.pad_token is None:
    student_tokenizer.pad_token = student_tokenizer.eos_token

def dialogsum_format(example):
    """
    Format DialogSum examples for summarization training.
    DialogSum has 'dialogue' and 'summary' fields.
    """
    dialogue = example['dialogue']
    summary = example['summary']
    
    # Create a conversation format for summarization
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes dialogues concisely and accurately."},
        {"role": "user", "content": f"Please summarize the following dialogue:\n\n{dialogue}"},
        {"role": "assistant", "content": summary}
    ]
    
    # Apply chat template
    text = student_tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    return {"text": text}

# Preprocess and tokenize the dataset
print("Preprocessing and tokenizing dataset...")
original_columns = train_dataset.column_names
train_dataset = train_dataset.map(dialogsum_format, remove_columns=original_columns)
val_dataset = val_dataset.map(dialogsum_format, remove_columns=original_columns)

def tokenize_function(examples):
    return student_tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=config["tokenizer"]["max_length"], 
        padding="max_length"
    )

train_tokenized = train_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
val_tokenized = val_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

tokenized_dataset = {"train": train_tokenized, "test": val_tokenized}

print(f"Dataset preparation complete. Train samples: {len(tokenized_dataset['train'])}, Test samples: {len(tokenized_dataset['test'])}")
print("Loading models...")

# Load models with configurable flash attention
model_kwargs = {"torch_dtype": torch.bfloat16}
if config["model_config"]["use_flash_attention"]:
    model_kwargs["attn_implementation"] = "flash_attention_2"

teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"], **model_kwargs)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_config, get_peft_model, LoraConfig

# --- a) Define your 8‑bit config
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

# --- b) Load the quantized student model
student_model = AutoModelForCausalLM.from_pretrained(
    config["models"]["student"],
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Disable kv-cache to save even more RAM:
student_model.config.use_cache = False

# --- c) Create a LoRA PEFT config
peft_config = LoraConfig(
    r=16,                      # rank of the LoRA update matrices
    lora_alpha=32,             # scaling
    target_modules=["q_proj", "v_proj"],  # which modules to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- d) Wrap your model with PEFT
student_model = get_peft_model(student_model, peft_config)

# 3. (Optional) Disable KV cache on the quantized model
student_model.config.use_cache = False



# Optionally freeze layers of the student model based on spectrum configuration
if "spectrum" in config and "layers_to_unfreeze" in config["spectrum"]:
    def freeze_student_spectrum(model, unfrozen_layers_file):
        with open(unfrozen_layers_file, 'r') as file:
            unfrozen_layers = yaml.safe_load(file)['unfrozen_parameters']
        
        total_params = sum(p.numel() for p in model.parameters())
        frozen_params = 0
        
        for name, param in model.named_parameters():
            if not any(layer in name for layer in unfrozen_layers):
                param.requires_grad = False
                frozen_params += param.numel()
            else:
                param.requires_grad = True
        
        print(f"Frozen {frozen_params}/{total_params} parameters ({100*frozen_params/total_params:.1f}%)")

    # Apply freezing to student model
    freeze_student_spectrum(student_model, config["spectrum"]["layers_to_unfreeze"])
else:
    print("Spectrum configuration not found. All layers of the student model will be trainable.")

def pad_logits(student_logits, teacher_logits):
    """
    Pad logits to match vocabulary sizes if different.
    """
    student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)
    if student_size != teacher_size:
        pad_size = abs(student_size - teacher_size)
        pad_tensor = torch.zeros(
            (*teacher_logits.shape[:-1], pad_size), 
            dtype=teacher_logits.dtype, 
            device=teacher_logits.device
        )
        if student_size < teacher_size:
            return torch.cat([student_logits, pad_tensor], dim=-1), teacher_logits
        else:
            return student_logits, torch.cat([teacher_logits, pad_tensor], dim=-1)
    return student_logits, teacher_logits

class LogitsTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Ensure teacher model is on the same device
        self.teacher_model = self.teacher_model.to(device)
        
        # Handle DataParallel/DistributedDataParallel
        student_model = model.module if hasattr(model, 'module') else model
        teacher_model = self.teacher_model.module if hasattr(self.teacher_model, 'module') else self.teacher_model

        # Get student outputs
               
        inputs.pop("labels", None)
        # Compute student causal‐LM loss
        student_outputs = model(**inputs, labels=inputs["input_ids"])
        
        # Get teacher outputs without gradients
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs, labels=inputs["input_ids"])

        # Compute distillation loss
        causal_loss = student_outputs.loss
        kd_loss = self.distillation_loss(
             student_logits=student_outputs.logits, 
             teacher_logits=teacher_outputs.logits, 
             causal_loss=causal_loss
        )
        custom_loss = kd_loss
        
        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def distillation_loss(self, student_logits, teacher_logits, causal_loss):
        """
        Compute knowledge distillation loss combining KL divergence and original loss.
        """
        device = student_logits.device
        student_logits, teacher_logits = pad_logits(student_logits.to(device), teacher_logits.to(device))
        
        # Scale logits by temperature
        temperature = config["distillation"]["temperature"]
        student_logits_scaled = student_logits / temperature
        teacher_logits_scaled = teacher_logits / temperature

        # Compute KL divergence loss
        loss_kd = F.kl_div(
            F.log_softmax(student_logits_scaled, dim=-1),
            F.softmax(teacher_logits_scaled, dim=-1),
            reduction='batchmean'
        ) * (temperature ** 2)

        # Combine losses
        alpha = config["distillation"]["alpha"]
        combined_loss = alpha * loss_kd + (1 - alpha) * causal_loss
        
        return combined_loss

# Training arguments
training_arguments = TrainingArguments(**config["training"],
                                       push_to_hub=True,
                                       push_to_hub_model_id="llama32-distilled-dialogsum",
                                        hub_strategy="end",  
                                        )

# Create the custom SFT Trainer
trainer = LogitsTrainer(
    model=student_model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    # tokenizer=student_tokenizer,
    args=training_arguments,
    callbacks=[SimpleCSVLoggerCallback("training_log.csv")],
)

# Add the teacher model to the trainer
trainer.teacher_model = teacher_model

print("Starting training...")
# Train the model
trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

# Save the final model
print(f"Saving model to {config['training']['output_dir']}")
trainer.save_model(config["training"]["output_dir"])
trainer.tokenizer.save_pretrained(config["training"]["output_dir"])

trainer.push_to_hub()


print("Training completed successfully!")