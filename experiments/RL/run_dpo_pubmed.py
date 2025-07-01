import os
from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth import is_bfloat16_supported
from datasets import load_dataset, Dataset
import torch
from trl import DPOTrainer, DPOConfig
import random
PatchDPOTrainer()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


max_seq_length = 1024

original_dataset = load_dataset("MothMalone/SLMS-KD-Benchmarks", "pubmedqa")

def create_dpo_dataset(examples):
    dpo_examples = []
    
    for i in range(len(examples['question'])):
        question = examples['question'][i]
        context = examples['context'][i]
        correct_answer = examples['long_answer'][i]
        final_decision = examples['final_decision'][i]
        
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        chosen = correct_answer
        
        if final_decision.lower() == 'yes':
            rejected = "No, based on the context provided, this is not supported."
        elif final_decision.lower() == 'no':
            rejected = "Yes, based on the context provided, this is clearly supported."
        else:
            # For 'maybe' cases, create a more definitive wrong answer
            rejected = "The answer is definitively yes without any uncertainty."
        
        dpo_examples.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected
        })
    
    return dpo_examples

train_dpo_data = create_dpo_dataset(original_dataset['train'])
test_dpo_data = create_dpo_dataset(original_dataset['test'])

train_dataset = Dataset.from_list(train_dpo_data)
test_dataset = Dataset.from_list(test_dpo_data)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
print("\nSample from train dataset:")
print(train_dataset[0])

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/zephyr-sft-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    max_seq_length=max_seq_length,
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=DPOConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        num_train_epochs=1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        seed=42,
        output_dir="outputs",
        beta=0.1,
        remove_unused_columns=False,
    ),
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    max_length=1024,
    max_prompt_length=512,
)

dpo_trainer.train()