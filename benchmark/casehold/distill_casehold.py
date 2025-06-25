import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import os
import json

def jensen_shannon_divergence(p_logits, q_logits):
    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    m = 0.5 * (p + q)
    # Use log_target=True for KLDivLoss when the target is in log space
    kl_pm = F.kl_div(F.log_softmax(p_logits, dim=-1), m, reduction='batchmean', log_target=False)
    kl_qm = F.kl_div(F.log_softmax(q_logits, dim=-1), m, reduction='batchmean', log_target=False)
    return 0.5 * (kl_pm + kl_qm)

class LocalDistillTrainer(Trainer):
    def __init__(self, teacher_model, temperature=2.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        # Ensure teacher is on the correct device and in eval mode
        self.teacher_model.to(self.model.device)
        self.teacher_model.eval()

    def pad_logits(self, student_logits, teacher_logits):
        """Pads the smaller logit tensor to match the larger one's vocab size."""
        s_vocab_size = student_logits.size(-1)
        t_vocab_size = teacher_logits.size(-1)

        if s_vocab_size == t_vocab_size:
            return student_logits, teacher_logits

        if s_vocab_size < t_vocab_size:
            pad_size = t_vocab_size - s_vocab_size
            padding = torch.zeros(*student_logits.shape[:-1], pad_size, device=student_logits.device)
            student_logits = torch.cat([student_logits, padding], dim=-1)
        else: # t_vocab_size < s_vocab_size
            pad_size = s_vocab_size - t_vocab_size
            padding = torch.zeros(*teacher_logits.shape[:-1], pad_size, device=teacher_logits.device)
            teacher_logits = torch.cat([teacher_logits, padding], dim=-1)
            
        return student_logits, teacher_logits

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Student forward pass
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # Teacher forward pass (no gradients needed)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        student_logits, teacher_logits = self.pad_logits(student_logits, teacher_logits)

        # KL Divergence distillation loss
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean',
            log_target=False 
        ) * (self.temperature ** 2)

        # JS Divergence loss for regularization
        js_loss = jensen_shannon_divergence(student_logits, teacher_logits)

        # Original cross-entropy loss from the student
        original_loss = student_outputs.loss

        # Combined loss
        custom_loss = self.alpha * kl_loss + (1 - self.alpha) * original_loss + 0.1 * js_loss

        return (custom_loss, student_outputs) if return_outputs else custom_loss
class LocalDistillTrainer(Trainer):
    def __init__(self, teacher_model, temperature=2.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_model.to(self.model.device)
        self.teacher_model.eval()

    def pad_logits(self, student_logits, teacher_logits):
        s_vocab_size = student_logits.size(-1)
        t_vocab_size = teacher_logits.size(-1)
        if s_vocab_size == t_vocab_size:
            return student_logits, teacher_logits
        if s_vocab_size < t_vocab_size:
            pad_size = t_vocab_size - s_vocab_size
            padding = torch.zeros(*student_logits.shape[:-1], pad_size, device=student_logits.device)
            student_logits = torch.cat([student_logits, padding], dim=-1)
        else:
            pad_size = s_vocab_size - t_vocab_size
            padding = torch.zeros(*teacher_logits.shape[:-1], pad_size, device=teacher_logits.device)
            teacher_logits = torch.cat([teacher_logits, padding], dim=-1)
        return student_logits, teacher_logits

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Manually calculate the student's original loss.
        labels = inputs.pop("labels", None) 
        
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        student_logits, teacher_logits = self.pad_logits(student_logits, teacher_logits)

        kl_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean',
            log_target=False 
        ) * (self.temperature ** 2)

        js_loss = jensen_shannon_divergence(student_logits, teacher_logits)


        # Manually calculate the student's original cross-entropy loss
        original_loss = 0.0
        if labels is not None:
            # Standard language modeling loss calculation
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            original_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


        custom_loss = self.alpha * kl_loss + (1 - self.alpha) * original_loss + 0.1 * js_loss
        
        if return_outputs:
            return {"loss": custom_loss, "logits": student_outputs.logits, "hidden_states": student_outputs.hidden_states}
        
        return custom_loss
def preprocess_casehold(tokenizer, max_length=2048):
    ds = load_dataset("MothMalone/SLMS-KD-Benchmarks", "casehold")
    data = ds['train']
    
    def format_example(example):
        prompt = (f"Citing Prompt: {example['citing_prompt']}\n\nChoices:\n"
                  f"0: {example['holding_0']}\n1: {example['holding_1']}\n"
                  f"2: {example['holding_2']}\n3: {example['holding_3']}\n"
                  f"4: {example['holding_4']}\n\nCorrect Answer Index: {example['label']}")
        return {"text": prompt}

    data = data.map(format_example)
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")

    tokenized = data.map(tokenize_fn, batched=True, num_proc=4, remove_columns=data.column_names)
    return tokenized.train_test_split(test_size=0.1, seed=42)

def main():
    TEACHER_MODEL_ID = "meta-llama/Llama-2-13b-hf"
    STUDENT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    OUTPUT_DIR = "student_model_casehold"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading teacher model (Llama-2-13B) in 4-bit precision...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_ID,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("Teacher model loaded.")

    print(f"Loading student model ({STUDENT_MODEL_ID})...")
    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    print("Student model loaded.")
    
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        student_model.resize_token_embeddings(len(tokenizer))
        teacher_model.resize_token_embeddings(len(tokenizer))

    print("Preprocessing dataset...")
    tokenized_dataset = preprocess_casehold(tokenizer)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        optim="adamw_8bit",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        save_steps=500,
        logging_steps=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        report_to="none",
    )

    trainer = LocalDistillTrainer(
        teacher_model=teacher_model,
        temperature=2.0,
        alpha=0.5,
        model=student_model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        args=training_args,
    )

    print("Starting distillation training on RTX 3090...")
    trainer.train()
    print("Training complete.")

    trainer.save_model(OUTPUT_DIR)
    print(f"Student model saved to {OUTPUT_DIR}")

    eval_metrics = trainer.evaluate()
    eval_metrics_path = os.path.join(OUTPUT_DIR, "eval_metrics.json")
    with open(eval_metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"Evaluation metrics saved to {eval_metrics_path}")

if __name__ == "__main__":
    main()