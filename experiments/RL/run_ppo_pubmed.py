import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from datasets import load_dataset, Dataset
import torch
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import numpy as np
from typing import List

max_seq_length = 1024

# Load dataset
dataset = load_dataset("MothMalone/SLMS-KD-Benchmarks", "pubmedqa")

def prepare_ppo_dataset(examples, max_samples=50):
    """
    Prepare dataset for PPO training with Unsloth
    """
    ppo_data = []
    
    for i in range(min(max_samples, len(examples['question']))):
        # Create concise prompts for PPO
        context = examples['context'][i][:300]  # Truncate context to save tokens
        question = examples['question'][i]
        
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        ppo_data.append({
            'query': prompt,
            'reference_answer': examples['long_answer'][i],
            'correct_decision': examples['final_decision'][i].lower()
        })
    
    return ppo_data

# Prepare training data
print("Preparing PPO dataset...")
train_data = prepare_ppo_dataset(dataset['train'], max_samples=30)
print(f"Created {len(train_data)} training examples")

# Load Unsloth model
print("Loading Unsloth model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/zephyr-sft-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

# Add LoRA adapters
print("Adding LoRA adapters...")
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

# Enable training mode
model = FastLanguageModel.for_training(model)

# Setup tokenizer for PPO
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Convert to PPO model with value head
print("Converting to PPO model...")
ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model,
    torch_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
    device_map="auto"
)

# PPO Configuration optimized for Unsloth
ppo_config = PPOConfig(
    model_name="unsloth/zephyr-sft-bnb-4bit",
    learning_rate=1e-5,
    batch_size=2,  # Small batch size for memory efficiency
    mini_batch_size=1,
    gradient_accumulation_steps=2,
    optimize_cuda_cache=True,
    early_stopping=False,
    target_kl=0.1,
    ppo_epochs=2,
    cliprange=0.2,
    cliprange_value=0.2,
    vf_coef=0.1,
    seed=42,
    log_with=None,  # Disable wandb
    remove_unused_columns=False,
)

# Create PPO trainer
print("Creating PPO trainer...")
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=ppo_model,
    tokenizer=tokenizer,
)

def compute_medical_qa_reward(generated_text: str, reference_answer: str, correct_decision: str) -> float:
    """
    Reward function tailored for medical Q&A
    """
    reward = 0.0
    generated_lower = generated_text.lower().strip()
    reference_lower = reference_answer.lower()
    
    # Length reward (prefer reasonable length responses)
    gen_length = len(generated_text.split())
    if 10 <= gen_length <= 80:
        reward += 1.0
    elif gen_length < 5:
        reward -= 1.5  # Heavy penalty for too short
    elif gen_length > 120:
        reward -= 0.8  # Penalty for too long
    
    # Content overlap reward
    gen_words = set(generated_lower.split())
    ref_words = set(reference_lower.split())
    
    if len(ref_words) > 0:
        # Jaccard similarity
        intersection = len(gen_words.intersection(ref_words))
        union = len(gen_words.union(ref_words))
        jaccard_score = intersection / union if union > 0 else 0
        reward += jaccard_score * 3.0
    
    # Decision alignment reward
    if correct_decision in generated_lower:
        reward += 1.5
    
    # Medical relevance keywords
    medical_terms = [
        'patient', 'study', 'clinical', 'treatment', 'evidence', 
        'research', 'therapy', 'diagnosis', 'medical', 'health',
        'disease', 'condition', 'outcome', 'trial', 'efficacy'
    ]
    
    medical_score = 0
    for term in medical_terms:
        if term in generated_lower:
            medical_score += 1
    
    reward += min(medical_score * 0.2, 1.0)  # Cap medical term bonus
    
    # Coherence bonus (simple check for complete sentences)
    if generated_text.strip().endswith('.'):
        reward += 0.3
    
    # Penalty for repetitive text
    words = generated_text.split()
    if len(words) > 10:
        unique_words = len(set(words))
        repetition_ratio = unique_words / len(words)
        if repetition_ratio < 0.7:
            reward -= 0.5
    
    return max(0.1, reward)  # Minimum reward to avoid zero

def train_unsloth_ppo():
    """
    Main training function for Unsloth PPO
    """
    generation_kwargs = {
        "min_length": -1,
        "top_k": 50,
        "top_p": 0.95,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 80,
        "temperature": 0.7,
    }
    
    print("Starting Unsloth PPO training...")
    
    for epoch in range(2):  # Start with few epochs
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch + 1}")
        print(f"{'='*50}")
        
        epoch_rewards = []
        
        # Process data in batches
        for batch_idx in range(0, len(train_data), ppo_config.batch_size):
            batch_end = min(batch_idx + ppo_config.batch_size, len(train_data))
            batch = train_data[batch_idx:batch_end]
            
            print(f"\nProcessing batch {batch_idx//ppo_config.batch_size + 1}...")
            
            try:
                # Prepare queries
                queries = [item['query'] for item in batch]
                query_tensors = []
                
                for query in queries:
                    # Tokenize with proper truncation
                    tokens = tokenizer(
                        query,
                        return_tensors="pt",
                        max_length=400,  # Leave room for generation
                        truncation=True,
                        padding=False
                    )["input_ids"].squeeze()
                    
                    if len(tokens.shape) == 0:
                        tokens = tokens.unsqueeze(0)
                    query_tensors.append(tokens)
                
                # Generate responses using Unsloth-optimized model
                print("Generating responses...")
                response_tensors = ppo_trainer.generate(
                    query_tensors,
                    return_prompt=False,
                    **generation_kwargs
                )
                
                responses = []
                for response_tensor in response_tensors:
                    response_text = tokenizer.decode(
                        response_tensor.squeeze(),
                        skip_special_tokens=True
                    ).strip()
                    responses.append(response_text)
                
                # Calculate rewards
                print("Computing rewards...")
                rewards = []
                for i, response in enumerate(responses):
                    reward = compute_medical_qa_reward(
                        response,
                        batch[i]['reference_answer'],
                        batch[i]['correct_decision']
                    )
                    rewards.append(reward)


                reward_tensors = [torch.tensor(r, dtype=torch.float32) for r in rewards]
                
                # PPO training step
                print("Performing PPO step...")
                stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
                
                # Track progress
                avg_reward = np.mean(rewards)
                epoch_rewards.extend(rewards)
                
                print(f"Batch {batch_idx//ppo_config.batch_size + 1} completed!")
                print(f"Average reward: {avg_reward:.3f}")
                print(f"Rewards: {[f'{r:.2f}' for r in rewards]}")
                
                # Show sample output
                if len(responses) > 0:
                    print(f"\nSample Query: {queries[0][:100]}...")
                    print(f"Sample Response: {responses[0][:150]}...")
                    print(f"Sample Reward: {rewards[0]:.3f}")
                
                print("-" * 50)
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Epoch summary
        if epoch_rewards:
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Average reward: {np.mean(epoch_rewards):.3f}")
            print(f"Reward std: {np.std(epoch_rewards):.3f}")
            print(f"Min/Max rewards: {np.min(epoch_rewards):.3f} / {np.max(epoch_rewards):.3f}")
    
    return ppo_trainer

# Run the training
if __name__ == "__main__":
    try:
        print("Starting Unsloth PPO Training Pipeline...")
        trained_ppo_trainer = train_unsloth_ppo()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("Saving trained model...")
        trained_ppo_trainer.model.save_pretrained("./unsloth_ppo_model")
        tokenizer.save_pretrained("./unsloth_ppo_model")
        
        print("Model saved to ./unsloth_ppo_model")
        
        print("\nTesting trained model...")
        test_query = "Context: Regular exercise has been shown to improve cardiovascular health.\n\nQuestion: Does exercise benefit heart health?\n\nAnswer:"
        
        # Tokenize test query
        test_tokens = tokenizer(test_query, return_tensors="pt")["input_ids"]
        
        with torch.no_grad():
            output = trained_ppo_trainer.model.generate(
                test_tokens,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(output[0][test_tokens.shape[1]:], skip_special_tokens=True)
        print(f"Test Query: {test_query}")
        print(f"Model Response: {response}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTrying fallback approach...")
        # Simplified fallback if full training fails
        print("Note: If you encounter memory issues, try reducing batch_size to 1 or using a smaller model.")