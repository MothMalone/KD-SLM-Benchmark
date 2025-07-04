from enum import Enum
import datasets
import os
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
from typing import Literal
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import transformers
import outlines
import torch
import json


from outlines.models.transformers import from_transformers

torch.cuda.empty_cache()
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")



def evaluate_model(
    model_name="meta-llama/Llama-3.2-1B",
    sample_size=100,
    dataset_path="MothMalone/SLMS-KD-Benchmarks",
    dataset_name="casehold",
    split=True,
    gpu_poor=True,
    quant_mode="4bit",
    output_dir="./results"
):
    """
    Evaluate a single model on Casehold dataset
    """
    
    print(f"🚀 Starting evaluation for: {model_name}")
    print(f"📊 Sample size: {sample_size}")
    print(f"💾 GPU Poor mode: {gpu_poor} ({quant_mode} quantization)")
    
    # Create output filenames
    model_short_name = model_name.split('/')[-1]
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    if gpu_poor:
        base_name = f"{model_short_name}_{quant_mode}_{timestamp}"
    else:
        base_name = f"{model_short_name}_full_{timestamp}"
    
    log_file = os.path.join(output_dir, f"metrics_{base_name}.csv")
    results_file = os.path.join(output_dir, f"detailed_{base_name}.csv")
    summary_file = os.path.join(output_dir, f"summary_{base_name}.txt")
    json_file = os.path.join(output_dir, f"results_{base_name}.json")
    
    print(f"📁 Results will be saved to: {output_dir}")
    
    # Load dataset
    print("📚 Loading dataset...")
    try:
        ds = datasets.load_dataset(dataset_path, dataset_name)
        if split:
            ds_split = 'test' if 'test' in ds else 'validation'
            ds = ds[ds_split].select(range(min(sample_size, len(ds[ds_split]))))
        else:
            ds = ds['train'].select(range(min(sample_size, len(ds['train']))))
        print(f"✅ Dataset loaded: {len(ds)} samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

    # Load model from transformers
    print("🤖 Loading model...")
    model_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if gpu_poor:
        if quant_mode == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:  # 8bit
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        print("✅ Hugging Face model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

    # Load tokenizer
    print("🔤 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if tokenizer.chat_template is None:
            print("Tokenizer missing chat template. Applying default Llama 3 template.")
            tokenizer.chat_template = (
                "{% for message in messages %}{% if message['role'] == 'system' %}"
                "{{'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}"
                "{% elif message['role'] == 'user' %}"
                "{{'<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}"
                "{% elif message['role'] == 'assistant' %}"
                "{{'<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>'}}"
                "{% endif %}{% endfor %}{% if add_generation_prompt %}"
                "{{'<|start_header_id|>assistant<|end_header_id|>\n\n'}}{% endif %}"
            )
        print("✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return None

    # Create outlines model by wrapping the loaded transformers model
    try:
        # FIX: Using the function call you requested.
        model = from_transformers(hf_model, tokenizer)
        print("✅ Outlines model created successfully")
    except Exception as e:
        print(f"❌ Error creating outlines model: {e}")
        return None

    # Initialize tracking
    y_true = []
    y_pred = []
    detailed_results = []
    metrics_history = []

    print("\n🔄 Starting evaluation...")

    
    # Process each sample
    for i, row in enumerate(ds):
        print(f"Processing sample {i+1}/{len(ds)}", end=" - ")
        
        # Create a simple prompt that matches the fine-tuning task.
        user_prompt = row['citing_prompt']
        chat = [{"role": "user", "content": user_prompt}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        
        # The choices for the model are the full text of the holdings.
        choices = [row[f'holding_{j}'] for j in range(5)]
        true_label_text = row[f"holding_{row['label']}"]

        try:
            # Constrain generation to the full text of the choices.
            generator = outlines.generate.choice(model, choices)
            predicted_label_text = generator(prompt, max_tokens=256)
            
            # FIX: Find the best matching holding using fuzzy matching
            from difflib import SequenceMatcher
            
            best_match_score = 0
            predicted_label = -1  # if no good match
            
            for j in range(5):
                # Calculate similarity between predicted text and each holding
                similarity = SequenceMatcher(None, predicted_label_text.lower().strip(), 
                                           row[f'holding_{j}'].lower().strip()).ratio()
                if similarity > best_match_score:
                    best_match_score = similarity
                    predicted_label = int(j)
            
            # Compare numerical labels
            true_label = int(row['label'])  # This should be 0-4
            y_true.append(true_label)
            y_pred.append(predicted_label)

            if predicted_label != -1:
                is_correct = (int(true_label) == int(predicted_label))
                if is_correct:
                    print(f"True: {true_label}, Pred: {predicted_label} (sim: {best_match_score:.3f}), ✅")
                else:
                    print(f"True: {true_label}, Pred: {predicted_label} (sim: {best_match_score:.3f}), ❌")

                
                detailed_results.append({
                    'sample_id': i,
                    'prompt': user_prompt,
                    'true_holding': int(true_label),
                    'predicted_holding': int(predicted_label),
                    'correct': is_correct
                })
            else:
                print("Bad generation, Skip example.")
                continue
        except Exception as e:
            print(f"❌ Error: {str(e)[:100]}...")
            y_true.append(true_label_text)
            y_pred.append("ERROR")
            detailed_results.append({'sample_id': i, 'prompt': user_prompt, 'true_holding': true_label_text, 'predicted_holding': 'ERROR', 'correct': False, 'error': str(e)})

        # Calculate metrics every 10 samples
        if (i + 1) % 10 == 0 or (i + 1) == len(ds):
            accuracy = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            metrics_row = {
                "model": model_name,
                "sample": i + 1,
                "accuracy": round(accuracy * 100, 3),
                "f1_macro": round(f1_macro * 100, 3),
                "f1_weighted": round(f1_weighted * 100, 3)
            }
            metrics_history.append(metrics_row)
            pd.DataFrame(metrics_history).to_csv(log_file, index=False)
            print(f"📊 Progress: Acc={accuracy*100:.1f}%")

    # --- Final calculations and reporting ---
    print("\n✅ Evaluation complete.")
    pd.DataFrame(detailed_results).to_csv(results_file, index=False)
    print(f"✅ Detailed results saved to: {results_file}")

    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"\n🎯 FINAL RESULTS FOR {model_name}")
    print("=" * 50)
    print(f"🎯 Accuracy: {accuracy * 100:.2f}%")
    print(f"📈 F1 Score (Weighted): {f1_weighted * 100:.2f}%")
    print(f"📈 F1 Score (Macro): {f1_macro * 100:.2f}%")
    
    all_labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    print(f"🔢 Confusion Matrix:\n{pd.DataFrame(cm, index=all_labels, columns=all_labels).to_string()}")

    # Create results dictionary
    results = {
        "model_name": model_name,
        "dataset": f"{dataset_path}/{dataset_name}",
        "total_samples": sample_size,
        "valid_samples": len(valid_pairs),
        "error_samples": sample_size - len(valid_pairs),
        "success_rate": len(valid_pairs) / sample_size * 100,
        "metrics": {
            "accuracy": float(accuracy * 100),
            "precision_weighted": float(precision * 100),
            "recall_weighted": float(recall * 100),
            "f1_weighted": float(f1 * 100),
            "f1_macro": float(f1_macro * 100)
        },
        "confusion_matrix": conf_matrix.tolist(),
        "label_distribution": {
            "true_labels": np.bincount(y_true_array, minlength=5).tolist(),  # 5 holdings
            "predicted_labels": np.bincount(y_pred_array, minlength=5).tolist()
        },
        "configuration": {
            "gpu_poor": gpu_poor,
            "quantization": quant_mode if gpu_poor else None,
            "evaluation_time": str(pd.Timestamp.now())
        }
    }

    # Save JSON results
    try:
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ JSON results saved to: {json_file}")
    except Exception as e:
        print(f"⚠️ Warning: Could not save JSON results: {e}")
    
    # Save text summary
    try:
        with open(summary_file, 'w') as f:
            f.write(f"=== CASEHOLD EVALUATION RESULTS ===\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Dataset: {dataset_path}/{dataset_name}\n")
            f.write(f"Total Samples: {sample_size}\n")
            f.write(f"Valid Predictions: {len(valid_pairs)}\n")
            f.write(f"Success Rate: {len(valid_pairs)/sample_size*100:.2f}%\n\n")
            
            f.write(f"=== PERFORMANCE METRICS ===\n")
            f.write(f"Accuracy: {accuracy * 100:.3f}%\n")
            f.write(f"Precision (Weighted): {precision * 100:.3f}%\n")
            f.write(f"Recall (Weighted): {recall * 100:.3f}%\n")
            f.write(f"F1 Score (Weighted): {f1 * 100:.3f}%\n")
            f.write(f"F1 Score (Macro): {f1_macro * 100:.3f}%\n\n")
            
            f.write(f"=== CONFUSION MATRIX ===\n")
            f.write(f"Rows: True Labels, Columns: Predicted Labels\n")
            f.write(f"Order: [holding_0=0, holding_1=1, holding_2=2, holding_3=3, holding_4=4]\n")
            f.write(f"{conf_matrix}\n\n")
            
            f.write(f"=== LABEL DISTRIBUTION ===\n")
            f.write(f"True Labels: {np.bincount(y_true_array, minlength=5).tolist()}\n")
            f.write(f"Predicted Labels: {np.bincount(y_pred_array, minlength=5).tolist()}\n\n")
            
            f.write(f"=== CONFIGURATION ===\n")
            f.write(f"GPU Poor Mode: {gpu_poor}\n")
            f.write(f"Quantization: {quant_mode if gpu_poor else 'None'}\n")
            f.write(f"Evaluation Time: {pd.Timestamp.now()}\n")
            
        print(f"✅ Summary saved to: {summary_file}")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not save summary: {e}")
    
    torch.cuda.empty_cache()
    return {"model_name": model_name, "metrics": {"accuracy": accuracy, "f1_macro": f1_macro, "f1_weighted": f1_weighted}}

def evaluate_all_models(
    models=None,
    sample_size=100,
    output_dir="./results",
    gpu_poor=False,
    quant_mode="4bit"
):
    """Evaluate multiple models sequentially on CaseHold dataset"""
    if models is None:
        models = [
            "bigscience/bloomz-560m",
            "facebook/opt-350m",
            "EleutherAI/pythia-410m"
        ]
    
    all_results = []
    
    print(f"🚀 Starting CaseHold evaluation of {len(models)} models")
    print("=" * 60)
    
    for i, model in enumerate(models, 1):
        print(f"\n🤖 [{i}/{len(models)}] Evaluating: {model}")
        print("-" * 40)
        
        try:
            result = evaluate_model(
                model_name=model,
                sample_size=sample_size,
                output_dir=output_dir,
                gpu_poor=gpu_poor,
                quant_mode=quant_mode
            )
            if result:
                all_results.append(result)
                print(f"✅ Successfully evaluated {model}")
            else:
                print(f"❌ Failed to evaluate {model}")
        except Exception as e:
            print(f"❌ Unhandled error evaluating {model}: {e}")
            continue
        
        torch.cuda.empty_cache()
        print(f"🧹 GPU memory cleared")
    
     # Create comparison summary
    if all_results:
        print(f"\n📊 CASEHOLD COMPARISON SUMMARY")
        print("=" * 80)
        
        comparison_data = []
        for result in all_results:
            comparison_data.append({
                'Model': result['model_name'].split('/')[-1],
                'Valid Samples': result['valid_samples'],
                'Success Rate': f"{result['success_rate']:.1f}%",
                'Accuracy': f"{result['metrics']['accuracy']:.2f}%",
                'F1 Weighted': f"{result['metrics']['f1_weighted']:.2f}%",
                'F1 Macro': f"{result['metrics']['f1_macro']:.2f}%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Save comparison
        try:
            comparison_file = os.path.join(output_dir, f"casehold_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
            comparison_df.to_csv(comparison_file, index=False)
            print(f"\n✅ Comparison saved to: {comparison_file}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save comparison: {e}")
    
    return all_results

if __name__ == "__main__":
    custom_models = [
        "bigscience/bloomz-560m",
        "facebook/opt-350m",
        "EleutherAI/pythia-410m"
    ]
    results = evaluate_all_models(models=custom_models, sample_size=1000)
