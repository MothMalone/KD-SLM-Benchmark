#!/usr/bin/env python
# coding=utf-8
#
# Evaluation script for fine-tuned models on the CaseHOLD dataset.
# This script's structure is aligned with the provided pubmedqa example.

import warnings
import os
from typing import Dict, Any

import click
import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Suppress warnings for a cleaner output
torch.cuda.empty_cache()
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def format_casehold_prompt(example: Dict[str, Any], tokenizer: AutoTokenizer) -> str:
    """
    Creates a standardized prompt for the CaseHOLD dataset using the model's chat template.

    Args:
        example: A dictionary representing one row from the dataset.
        tokenizer: The model's tokenizer to apply the chat template.

    Returns:
        A formatted string prompt for the model.
    """
    user_prompt = example.get("citing_prompt", "")
    
    # Apply the chat template to format the input exactly as it was during training
    chat = [{"role": "user", "content": user_prompt}]
    prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    return prompt


@click.command()
@click.option('--gpu-poor', default=True, help='Run with quantization to save memory.')
@click.option('--quant-mode', type=click.Choice(['4bit', '8bit'], case_sensitive=False), default='4bit', help='Quantization mode: 4bit or 8bit.')
@click.option('--sample-size', default=1000, help='Number of samples to evaluate.')
@click.option('--dataset-path', default='MothMalone/SLMS-KD-Benchmarks', help="Dataset repository on Hugging Face Hub.")
@click.option('--dataset-name', default='casehold', help="Configuration name of the dataset.")
@click.option('--split', default='train', help='The dataset split to use for evaluation (e.g., train, validation, test).')
@click.option('--model-name', default='results-llama3.2-1b-casehold-sft', help='The model to evaluate (local path or Hub ID).')
def main(model_name, sample_size, dataset_path, dataset_name, split, gpu_poor, quant_mode):
    """
    Main function to run the evaluation pipeline.
    """
    # --- 1. Setup and Configuration ---
    print(f"GPU Poor Mode: {gpu_poor}")
    if gpu_poor:
        log_file_name = f"metrics_log_{model_name.replace('/', '_')}_{quant_mode}.csv"
    else:
        log_file_name = f"metrics_log_{model_name.replace('/', '_')}.csv"
    print(f"All metrics will be logged to file: {log_file_name}")

    # --- 2. Load Dataset ---
    try:
        ds = datasets.load_dataset(
            dataset_path, name=dataset_name, split=f"{split}[:{sample_size}]"
        )
        print(f"Successfully loaded {len(ds)} samples from '{dataset_name}' - split '{split}'.")
    except Exception as e:
        print(f"Failed to load dataset. Error: {e}")
        return

    # --- 3. Load Model and Tokenizer ---
    quantization_config = None
    if gpu_poor:
        if quant_mode == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else: # 8bit
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # ===================================================================
        # FIX: Manually set chat template if it's missing
        # ===================================================================
        if tokenizer.chat_template is None:
            print("Tokenizer missing chat template. Applying default Llama 3 template.")
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "{{ '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
                "{% elif message['role'] == 'user' %}"
                "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
                "{% elif message['role'] == 'assistant' %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
                "{% endif %}"
            )
        # ===================================================================

    except Exception as e:
        print(f"Failed to load model or tokenizer from {model_name}. Error: {e}")
        return

    # --- 4. Evaluation Loop ---
    y_true = []
    y_pred = []
    metrics_log = []
    
    model.eval()
    print("\nStarting evaluation loop...")
    with torch.no_grad():
        for i, row in enumerate(ds):
            # Prepare the prompt and the ground truth answer
            prompt = format_casehold_prompt(row, tokenizer)
            ground_truth_answer = row[f"holding_{row['label']}"]

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
            
            # Generate the model's prediction
            output = model.generate(
                **inputs,
                max_new_tokens=128,  # Allow enough tokens for a legal holding
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )
            
            # Decode the full output and then isolate the generated part
            full_text = tokenizer.decode(output[0], skip_special_tokens=True)
            # The generated text is what comes after the prompt
            # We must remove the prompt text to get only the model's generation
            prompt_for_decode = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
            predicted_answer = full_text.replace(prompt_for_decode, "")

            y_true.append(ground_truth_answer)
            y_pred.append(predicted_answer.strip())

            # Log metrics periodically
            if (i + 1) % 100 == 0 or (i + 1) == len(ds):
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
                f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

                metrics_row = {
                    "Accuracy": round(accuracy * 100, 3),
                    "Precision": round(precision * 100, 3),
                    "Recall": round(recall * 100, 3),
                    "F1 Score": round(f1 * 100, 3),
                    "Macro F1 Score": round(f1_macro * 100, 3),
                    "Weighted F1 Score": round(f1_weighted * 100, 3),
                }
                metrics_log.append(metrics_row)
                
                pd.DataFrame(metrics_log).to_csv(log_file_name, index=False)
                print(f"  ... evaluated and logged metrics after {i + 1} cases.")

    print("Evaluation loop complete.")

    # --- 5. Final Metrics Calculation and Report ---
    print("\n--- Final Evaluation Results ---")
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    all_labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    print(f"Final Accuracy: {accuracy * 100:.2f}%")
    print(f"Final Macro Precision: {precision_macro * 100:.2f}%")
    print(f"Final Macro Recall: {recall_macro * 100:.2f}%")
    print(f"Final Macro F1 Score: {f1_macro * 100:.2f}%")
    print(f"Final Weighted F1 Score: {f1_weighted * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=all_labels, columns=all_labels))
    print("\nEvaluation complete. Final metrics saved.")


if __name__ == "__main__":
    main()
