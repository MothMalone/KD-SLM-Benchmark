import os
from huggingface_hub import login
import vllm
import click
import wandb
import transformers
import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.environ['HF_TOKEN']
WANDB_API_KEY = os.environ['WANDB_API_KEY']
login(HF_TOKEN)
wandb.login(WANDB_API_KEY)
torch.cuda.empty_cache()
transformers.logging.set_verbosity_info()

@click.command()
@click.option('--dataset-name', default='MothMalone/SLMS-KD-Benchmarks', help='HuggingFace dataset identifier')
@click.option('--dataset-config', default='pubmedqa', help='Dataset config name')
@click.option('--teacher-model', default='meta-llama/Llama-3.2-3B', help='Teacher model identifier')
@click.option('--student-model', default='meta-llama/Llama-3.2-1B', help='Student model identifier')
@click.option('--max-length', default=4096, type=int, help='Max token length')
@click.option('--epochs', default=10, type=int, help='Number of training epochs')
@click.option('--batch-size', default=1, type=int, help='Per-device train batch size')
@click.option('--gradient-accumulation-steps', default=8, type=int, help='Gradient accumulation steps')
@click.option('--save-steps', default=1000, type=int, help='Save checkpoint every N steps')
@click.option('--logging-steps', default=1, type=int, help='Logging every N steps')
@click.option('--learning-rate', default=2e-5, type=float, help='Learning rate')
@click.option('--weight-decay', default=0.05, type=float, help='Weight decay')
@click.option('--warmup-ratio', default=0.1, type=float, help='Warmup ratio')
@click.option('--lr-scheduler-type', default='cosine', help='LR scheduler type')
@click.option('--resume-checkpoint', default=None, help='Checkpoint to resume from')
@click.option('--fp16/--no-fp16', default=False, help='Enable FP16 training')
@click.option('--bf16/--no-bf16', default=False, help='Enable BF16 training')
@click.option('--use-flash-attention/--no-flash-attention', default=False, help='Use Flash Attention')
@click.option('--output-dir', default='./results', help='Directory to save results')
def main(
    dataset_name, dataset_config,
    teacher_model, student_model,
    max_length, epochs, batch_size,
    gradient_accumulation_steps, save_steps,
    logging_steps, learning_rate,
    weight_decay, warmup_ratio,
    lr_scheduler_type, resume_checkpoint,
    fp16, bf16, use_flash_attention,
    output_dir
):
    wandb.init(project="KD-SLM")

    config = {
        "dataset": {"name": dataset_name, "config": dataset_config},
        "models": {"teacher": teacher_model, "student": student_model},
        "tokenizer": {"max_length": max_length},
        "distillation": {"temperature": 2.0, "alpha": 0.5},
        "training": {
            "report_to": "wandb",
            "output_dir": output_dir,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "save_steps": save_steps,
            "logging_steps": logging_steps,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "lr_scheduler_type": lr_scheduler_type,
            "resume_from_checkpoint": resume_checkpoint,
            "fp16": fp16,
            "bf16": bf16
        },
        "model_config": {"use_flash_attention": use_flash_attention}
    }

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    ds = load_dataset(config['dataset']['name'], config['dataset']['config'])['train']
    student_tokenizer = AutoTokenizer.from_pretrained(config['models']['student'])
    student_tokenizer.pad_token = student_tokenizer.eos_token

    def format_example(row):
        return {"question": f"\n{row['question']}"}

    ds = ds.map(format_example)
    ds = ds.remove_columns(['pubid','context','long_answer'])
    tokenized = ds.map(
        lambda x: student_tokenizer(x['question'], truncation=True, padding='max_length', max_length=max_length),
        batched=True, remove_columns=['question']
    )
    split = tokenized.train_test_split(test_size=0.1, seed=42)

    student = AutoModelForCausalLM.from_pretrained(
        student_model,
        quantization_config=bnb_config,
        device_map='auto'
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model,
        quantization_config=bnb_config,
        device_map='auto'
    )

    # Prepare for PEFT
    student = prepare_model_for_kbit_training(student)
    lora_conf = LoraConfig(
        r=16, lora_alpha=8,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM"
    )
    student = get_peft_model(student, lora_conf)

    class LogitsTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, 'to')}
            teacher.to(device)
            outputs_s = model(**inputs)
            with torch.no_grad():
                outputs_t = teacher(**inputs)
            s_logit, t_logit = outputs_s.logits, outputs_t.logits
            if s_logit.size(-1)!=t_logit.size(-1):
                diff = abs(s_logit.size(-1) - t_logit.size(-1))
                pad = torch.zeros(*t_logit.shape[:-1], diff, device=t_logit.device)
                if s_logit.size(-1)<t_logit.size(-1): s_logit = torch.cat([s_logit,pad], dim=-1)
                else: t_logit = torch.cat([t_logit,pad], dim=-1)
            S = s_logit / config['distillation']['temperature']
            T = t_logit / config['distillation']['temperature']
            loss_kd = F.kl_div(
                F.log_softmax(S, dim=-1),
                F.softmax(T, dim=-1),
                reduction='batchmean'
            ) * (config['distillation']['temperature']**2) / max_length
            return config['distillation']['alpha']*loss_kd + (1-config['distillation']['alpha'])*outputs_s.loss

    args = TrainingArguments(**config['training'])

    trainer = LogitsTrainer(
        model=student,
        train_dataset=split['train'],
        eval_dataset=split['test'],
        args=args
    )
    trainer.teacher_model = teacher
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(output_dir)

if __name__ == '__main__':
    main()
