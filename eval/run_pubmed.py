from enum import Enum
from openai import OpenAI
import datasets
import dotenv
import os
import warnings
import instructor
from pydantic import BaseModel, Field, constr, ValidationError, field_validator
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
import click
from typing import Literal
from huggingface_hub import login
from traitlets import default
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
import outlines
import torch
import wandb
from dotenv import load_dotenv
torch.cuda.empty_cache()
transformers.logging.set_verbosity_info()
warnings.filterwarnings("ignore")
load_dotenv('../.env')
# wandb.login(key = os.environ['WANDB_API_KEY'])
# login(os.environ['HF_TOKEN'])

class PubmedqaAnswer(Enum):
    yes = "yes"
    no = "no"
    maybe = "maybe"


@click.command()
@click.option('--gpu-poor', default = True, help = 'Run with quantization or not')
@click.option('--quant-mode', type=click.Choice(['4bit','8bit'], case_sensitive=False),default='4bit',help='Quantization mode: 4bit, 8bit or full fp16')
@click.option('--sample-size', default = 5, help = 'How many sample to run, default is 5 to make it easier to test')
@click.option('--dataset-path', default = 'MothMalone/SLMS-KD-Benchmarks', help = "Dataset to evaluate")
@click.option('--dataset-name', default = 'pubmedqa', help = "Dataset to evaluate")
@click.option('--split', default = True, help = 'If the dataset is split into train, val, test set')
@click.option('--model-name', default = 'meta-llama/Llama-3.2-1B', help = 'The model to evaluate')
def main(model_name, sample_size, dataset_path, dataset_name, split ,gpu_poor, quant_mode):
    print("GPU Poor", gpu_poor)
    if gpu_poor:
        log_file_name = f"metrics_log_{model_name.split('/')[1]}_{quant_mode}.csv"
    else:
        log_file_name = f"metrics_log_{model_name.split('/')[1]}.csv"
    print(f'All metrics will be logged to file {log_file_name}')


    ds = datasets.load_dataset(dataset_path, dataset_name)
    if split:
        ds = ds['test'].select(range(sample_size))
    else:
        ds = ds.select(range(sample_size))
    if gpu_poor:
        if quant_mode == "4bit":
            bits_and_bytes_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only = True, quantization_config = bits_and_bytes_config ,device_map = "auto") 
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                local_files_only=True,
                load_in_8bit=True,
                device_map="auto",
            )
    else: 
        print("Running in full precision")
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, device_map = "auto") 
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, device_map = "auto")        
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, device_map = "auto")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id    


    model =  outlines.from_transformers(model, tokenizer)
    label_mapping = {
        "yes": 1,
        "no": 0,
        "maybe": 2
    }
    def encode_labels(label):
        return label_mapping.get(label, -1)

    y_true = []
    y_pred = []
    metrics_df = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1 Score", "Macro F1 Score", "Invalid Cases"])

    for i, row in enumerate(ds):
        prompt = f"""
            Question: {row['question']}
            Context: {row['context']}
            Long Answer: {row['long_answer']}
            Give me the final decision: yes, no or maybe
        """
        predicted_label = model(prompt, PubmedqaAnswer)
        y_true.append(encode_labels(row['final_decision']))
        y_pred.append(encode_labels(predicted_label))
        if (i + 1) % 10 == 0 or (i + 1) == len(ds):
            y_true_array = np.array(y_true, dtype=int)
            y_pred_array = np.array(y_pred, dtype=int)
            accuracy = accuracy_score(y_true_array, y_pred_array)
            precision = precision_score(y_true_array, y_pred_array, average='weighted', zero_division=0)
            recall = recall_score(y_true_array, y_pred_array, average='weighted', zero_division=0)
            f1 = f1_score(y_true_array, y_pred_array, average='weighted', zero_division=0)
            f1_macro = f1_score(y_true_array, y_pred_array, average='macro', zero_division=0)
            conf_matrix = confusion_matrix(y_true_array, y_pred_array)

            metrics_row = pd.DataFrame([{
                "Accuracy": round(accuracy * 100, 3),
                "Precision": round(precision * 100, 3),
                "Recall": round(recall * 100, 3),
                "F1 Score": round(f1 * 100, 3),
                "Macro F1 Score": round(f1_macro * 100, 3),   
            }])
            metrics_df = pd.concat([metrics_df, metrics_row], ignore_index=True)
            
            metrics_df.to_csv(log_file_name, index=False, mode='w')
            print(f"Logged metrics after {i + 1} cases.")

        
    y_true_array = np.array(y_true, dtype=int)
    y_pred_array = np.array(y_pred, dtype=int)

    accuracy = accuracy_score(y_true_array, y_pred_array)
    precision = precision_score(y_true_array, y_pred_array, average='weighted', zero_division=0)
    recall = recall_score(y_true_array, y_pred_array, average='weighted', zero_division=0)
    f1 = f1_score(y_true_array, y_pred_array, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true_array, y_pred_array, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(y_true_array, y_pred_array)

    print(f"Final Accuracy: {accuracy * 100:.2f}%")
    print(f"Final Precision: {precision * 100:.2f}%")
    print(f"Final Recall: {recall * 100:.2f}%")
    print(f"Final F1 Score: {f1 * 100:.2f}%")
    print(f"Final Macro F1 Score: {f1_macro * 100:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)
        

if __name__ == "__main__":
    main()