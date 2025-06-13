import grpc
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import pickle
import base64
import logits_service_pb2_grpc # type: ignore
from logits_service_pb2 import LogitsRequest # type: ignore
from typing import Optional
import os

class RemoteTeacherClient:
    def __init__(self, server_address: str = "localhost:50051"):
        self.channel = grpc.insecure_channel(server_address, options=[('grpc.max_send_message_length', 500 * 1024 * 1024), ('grpc.max_receive_message_length', 500 * 1024 * 1024)])
        self.stub = logits_service_pb2_grpc.LogitsServiceStub(self.channel)
        
    def serialize_tensor(self, tensor: torch.Tensor) -> str:
        """Serialize tensor to base64 string"""
        tensor_bytes = pickle.dumps(tensor.cpu())
        return base64.b64encode(tensor_bytes).decode('utf-8')

    def deserialize_tensor(self, tensor_str: str) -> torch.Tensor:
        """Deserialize base64 string to tensor"""
        tensor_bytes = base64.b64decode(tensor_str.encode('utf-8'))
        return pickle.loads(tensor_bytes)

    def get_teacher_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Optional[torch.Tensor]:
        """Get logits from remote teacher model"""
        try:
            request = LogitsRequest(
                input_ids=self.serialize_tensor(input_ids),
                attention_mask=self.serialize_tensor(attention_mask),
                model_config="{}"
            )
            
            response = self.stub.GetLogits(request)
            
            if response.success:
                return self.deserialize_tensor(response.logits)
            else:
                print(f"Error from teacher: {response.error_message}")
                return None
                
        except Exception as e:
            print(f"Connection error: {e}")
            return None

    def close(self):
        self.channel.close()

class DistributedLogitsTrainer(SFTTrainer):
    def __init__(self, teacher_client: RemoteTeacherClient, temperature: float = 2.0, alpha: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_client = teacher_client
        self.temperature = temperature
        self.alpha = alpha

    def pad_logits(self, student_logits, teacher_logits):
        """Pad logits to match vocabulary sizes"""
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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Get student outputs
        student_outputs = model(**inputs)
        
        # Get teacher logits from remote server
        teacher_logits = self.teacher_client.get_teacher_logits(
            inputs['input_ids'], 
            inputs['attention_mask']
        )
        
        if teacher_logits is not None:
            # Compute distillation loss
            custom_loss = self.distillation_loss(
                student_outputs.logits, 
                teacher_logits.to(device), 
                student_outputs.loss
            )
        else:
            # Fallback to original loss if teacher is unavailable
            print("Warning: Teacher model unavailable, using only student loss")
            custom_loss = student_outputs.loss
        
        return (custom_loss, student_outputs) if return_outputs else custom_loss

    def distillation_loss(self, student_logits, teacher_logits, original_loss):
        """Compute knowledge distillation loss"""
        device = student_logits.device
        student_logits, teacher_logits = self.pad_logits(
            student_logits, teacher_logits.to(device)
        )
        
        # Scale logits by temperature
        student_logits_scaled = student_logits / self.temperature
        teacher_logits_scaled = teacher_logits / self.temperature

        # Compute KL divergence loss
        loss_kd = F.kl_div(
            F.log_softmax(student_logits_scaled, dim=-1),
            F.softmax(teacher_logits_scaled, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # Combine losses
        return self.alpha * loss_kd + (1 - self.alpha) * original_loss

def train_distributed_student():
    """Train student model with remote teacher"""
    
    # Configuration
    config = {
        "models": {
            "teacher_server": "34.87.113.245:50051",  # Replace with actual IP
            "student": "meta-llama/Llama-3.2-1B"
        },
        "dataset": {
            "name": "mlabonne/FineTome-100k",
            "split": "train",
            "seed": 42
        },
        "tokenizer": {
            "max_length": 4096,
            "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        },
        "training": {
            "output_dir": "./results",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "save_steps": 1000,
            "logging_first_step": True,
            "logging_steps": 1,
            "learning_rate": 2e-5,
            "weight_decay": 0.05,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine",
            "bf16": True, 
            "disable_tqdm": False,
        },
        "distillation": {
            "temperature": 2.0,
            "alpha": 0.5
        }
    }
    
    # Initialize remote teacher client
    teacher_client = RemoteTeacherClient(config["models"]["teacher_server"])
    
    # Load dataset
    dataset = load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])
    # dataset = dataset.select(range(10))
    dataset = dataset.shuffle(seed=config["dataset"]["seed"])
    
    # Load student model and tokenizer
    student_model = AutoModelForCausalLM.from_pretrained(
        config["models"]["student"],
        torch_dtype=torch.bfloat16)
    
        
    student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])
    student_tokenizer.chat_template = config["tokenizer"]["chat_template"]
    if not student_tokenizer.pad_token:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    # Preprocess dataset (same as original)
    def sharegpt_format(example):
        conversations = example['conversations']
        message = []
        
        if isinstance(conversations, list):
            for conversation in conversations:
                if isinstance(conversation, dict):
                    if conversation.get('from') == 'human':
                        message.append({"role": "user", "content": conversation.get('value', '')})
                    elif conversation.get('from') == 'gpt':
                        message.append({"role": "assistant", "content": conversation.get('value', '')})
                    elif conversation.get('from') == 'system':
                        message.insert(0, {"role": "system", "content": conversation.get('value', '')})

        if not any(msg.get('role') == 'system' for msg in message):
            message.insert(0, {"role": "system", "content": "You are a helpful assistant."})

        text = student_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        return {"text": text}

    # Process dataset
    original_columns = dataset.column_names
    dataset = dataset.map(sharegpt_format, remove_columns=original_columns)

    def tokenize_function(examples):
        return student_tokenizer(examples["text"], truncation=True, max_length=4096, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=8, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    
    # Create distributed trainer
    from transformers import TrainingArguments
    training_args = TrainingArguments(**config["training"])
    
    trainer = DistributedLogitsTrainer(
        teacher_client=teacher_client,
        temperature=config["distillation"]["temperature"],
        alpha=config["distillation"]["alpha"],
        model=student_model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        args=training_args,
    )
    
    try:
        # Train the model
        trainer.train()
        
        # Save the final model
        trainer.save_model(config["training"]["output_dir"])
        
    finally:
        # Clean up
        teacher_client.close()

if __name__ == "__main__":
    train_distributed_student()