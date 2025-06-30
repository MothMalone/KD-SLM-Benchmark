import os
import wandb
import transformers
import torch
import sys
import torch.nn.functional as F
import click
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

torch.cuda.empty_cache()
transformers.logging.set_verbosity_info()

config = {
    "dataset": {
        "name": "MothMalone/SLMS-KD-Benchmarks"
    }, 
    "models": {
        "teacher": "meta-llama/Llama-3.2-3B",
        "student": "meta-llama/Llama-3.2-1B"
    },
    "tokenizer": {
        "max_length": 4096,
        "chat_template": """
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
        "num_train_epochs": 1,
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

def get_quantization_config(quantization_type):
    """Returns the appropriate BitsAndBytesConfig based on quantization type"""
    if quantization_type == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif quantization_type == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
        )
    elif quantization_type == "4bit-fp4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif quantization_type == "4bit-nf4-bf16":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        return None

def sharegpt_format(row, tokenizer, chat_template):
    question = f"""
        Question: {row['question']}
        Context: {row['context']}
        Long Answer: {row['long_answer']}
        Give me the final decision: yes, no or maybe
        """
    
    messages = [
        {"role": "system", "content": "You are a helpful medical assistant. Answer questions based on the provided context and information."},
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": row['final_decision']}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=False  
    )
    
    return {"text": text}

def pad_logits(student_logits, teacher_logits):
    student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)
    if student_size != teacher_size:
        pad_size = abs(student_size - teacher_size)
        pad_tensor = torch.zeros((*teacher_logits.shape[:-1], pad_size), dtype=teacher_logits.dtype, device=teacher_logits.device)
        return (torch.cat([student_logits, pad_tensor], dim=-1), teacher_logits) if student_size < teacher_size else (student_logits, torch.cat([teacher_logits, pad_tensor], dim=-1))
    return student_logits, teacher_logits


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

@click.command()
@click.option('--quantization', 
              type=click.Choice(['none', '4bit', '8bit', '4bit-fp4', '4bit-nf4-bf16'], case_sensitive=False),
              default='none',
              help='Quantization method: none (full precision), 4bit (NF4), 8bit, 4bit-fp4, 4bit-nf4-bf16')
@click.option('--output-dir', default='./results', 
              help='Output directory for saving the trained model.')
@click.option('--epochs', default=1, type=int,
              help='Number of training epochs.')
@click.option('--batch-size', default=1, type=int,
              help='Per device training batch size.')
@click.option('--learning-rate', default=2e-5, type=float,
              help='Learning rate for training.')
@click.option('--teacher-model', default=None,
              help='Teacher model name/path (overrides config default)')
@click.option('--student-model', default=None,
              help='Student model name/path (overrides config default)')
@click.option('--temperature', default=2.0, type=float,
              help='Distillation temperature')
@click.option('--alpha', default=0.5, type=float,
              help='Distillation loss weight (0-1)')
def main(quantization, output_dir, epochs, batch_size, learning_rate, 
         teacher_model, student_model, temperature, alpha):
    quantization_desc = {
        'none': 'full precision',
        '4bit': '4-bit NF4 quantization',
        '8bit': '8-bit quantization', 
        '4bit-fp4': '4-bit FP4 quantization',
        '4bit-nf4-bf16': '4-bit NF4 with bfloat16'
    }
    
    click.echo(f"Starting training with {quantization_desc[quantization]}")
    click.echo(f"Temperature: {temperature}, Alpha: {alpha}")
    
    config["training"]["output_dir"] = output_dir
    config["training"]["num_train_epochs"] = epochs
    config["training"]["per_device_train_batch_size"] = batch_size
    config["training"]["learning_rate"] = learning_rate
    config["distillation"]["temperature"] = temperature
    config["distillation"]["alpha"] = alpha
    
    if teacher_model:
        config["models"]["teacher"] = teacher_model
        click.echo(f"Using custom teacher model: {teacher_model}")
    if student_model:
        config["models"]["student"] = student_model
        click.echo(f"Using custom student model: {student_model}")
    
    wandb.login()
    wandb.init(project="KD-SLM", config={
        "quantization": quantization,
        "output_dir": output_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "temperature": temperature,
        "alpha": alpha,
        "teacher_model": config["models"]["teacher"],
        "student_model": config["models"]["student"]
    })
    
    click.echo("Loading dataset...")
    dataset = load_dataset(config['dataset']['name'], 'pubmedqa')
    dataset = dataset['train']
        
    student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])
    student_tokenizer.chat_template = config["tokenizer"]["chat_template"]
    teacher_tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"])

    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    def tokenize_function(examples):
        return student_tokenizer(examples["text"], truncation=True, max_length=config["tokenizer"]["max_length"], padding="max_length")

    click.echo("Processing dataset...")
    dataset = dataset.map(lambda row: sharegpt_format(row, student_tokenizer, config["tokenizer"]["chat_template"]))
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != 'text'])
    dataset = dataset.map(tokenize_function, batched=True, num_proc=8, remove_columns=["text"])
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    click.echo(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    model_kwargs = {"torch_dtype": torch.float16}
    if config["model_config"]["use_flash_attention"]:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    quantization_config = get_quantization_config(quantization)
    use_quantization = quantization != 'none'
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
        click.echo(f"Quantization enabled: {quantization}")
    else:
        click.echo("Running in full precision mode")
    
    click.echo(f"Loading student model: {config['models']['student']}")
    student_model = AutoModelForCausalLM.from_pretrained(
        config["models"]["student"], 
        **model_kwargs
    )

    click.echo(f"Loading teacher model: {config['models']['teacher']}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config["models"]["teacher"], 
        **model_kwargs
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    if use_quantization:
        student_model = prepare_model_for_kbit_training(student_model)
    student_model = get_peft_model(student_model, lora_config)

    training_arguments = TrainingArguments(**config["training"])

    trainer = LogitsTrainer(
        model=student_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_arguments,
    )

    trainer.teacher_model = teacher_model
    
    click.echo("Starting training...")
    trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])
    
    click.echo(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    
    click.echo("Training completed successfully!")

if __name__ == "__main__":
    main()