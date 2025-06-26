from openai import OpenAI
import datasets
import dotenv
import os
import warnings
import concurrent.futures
import instructor
from pydantic import BaseModel, Field, validator
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import pandas as pd
import json
import re
from typing import List, Dict, Any, Union
import random
import csv

warnings.filterwarnings("ignore")

class FinQAResponse(BaseModel):
    reasoning: str = Field(
        description="Step‑by‑step chain‑of‑thought showing how the calculation was done."
    )
    answer: str = Field(
        description="The final numerical or text answer, no extra formatting."
    )

    @validator('answer', pre=True)
    def convert_answer_to_string(cls, v):
        """Convert any numeric answer to string format"""
        if isinstance(v, (int, float)):
            return str(v)
        return str(v) if v is not None else ""

    class Config:
        pass

def normalize_answer(answer_text):
    """
    Normalize answer text to extract just the numerical value
    """
    if not answer_text or answer_text in [None, "None", ""]:
        return None
    
    # Convert to string if it's not already
    answer_str = str(answer_text).strip()
    
    # Handle multiple values in one answer (take the first one)
    if " and " in answer_str:
        answer_str = answer_str.split(" and ")[0]
    
    # Remove common prefixes/suffixes
    answer_str = answer_str.replace("$", "").replace(",", "")
    
    # Extract percentage
    if "%" in answer_str:
        # Extract number before %
        match = re.search(r'([\d.-]+)%?', answer_str)
        if match:
            return match.group(1)
    
    # Extract millions notation
    if "million" in answer_str.lower():
        # Look for number before "million"
        match = re.search(r'([\d.-]+)\s*million', answer_str.lower())
        if match:
            return match.group(1)
    
    # Extract basic number (with potential decimal)
    match = re.search(r'-?[\d,]+\.?\d*', answer_str)
    if match:
        return match.group(0).replace(",", "")
    
    # If no number found, return the original (might be text answer)
    return answer_str

def is_answer_parsable(answer):
    """Check if an answer is parsable (can be normalized to a valid value)"""
    normalized = normalize_answer(answer)
    if normalized is None:
        return False
    
    # Try to convert to float to check if it's a valid number
    try:
        float(normalized)
        return True
    except (ValueError, TypeError):
        # If it's not a number, consider it parsable if it's a non-empty string
        return len(str(normalized).strip()) > 0

def are_answers_equivalent(predicted, ground_truth, tolerance=0.01):
    """
    Check if predicted answer matches ground truth within tolerance
    """
    # Normalize both answers
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    
    if pred_norm is None or gt_norm is None:
        return False
    
    try:
        # Try to convert to float for numerical comparison
        pred_float = float(pred_norm)
        gt_float = float(gt_norm)
        
        # Check if they're close enough (within tolerance)
        if abs(gt_float) < 1e-10:  # Ground truth is essentially zero
            return abs(pred_float) < tolerance
        else:
            return abs(pred_float - gt_float) / abs(gt_float) < tolerance
    
    except (ValueError, TypeError):
        # If conversion fails, do string comparison
        return pred_norm.lower().strip() == gt_norm.lower().strip()

# Load environment variables
dotenv.load_dotenv('../../.env')

# Load FinQA dataset
print("Loading FinQA dataset...")
ds = datasets.load_dataset("MothMalone/SLMS-KD-Benchmarks", "finqa")

# Use train and validation splits (not test as specified)
train_data = ds['train']
val_data = ds['validation']

print(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples.")

DATASET_CONTEXT = """
FinQA is a financial question answering dataset that requires numerical reasoning over financial documents.
Each question is based on financial reports containing tables and text. You need to analyze the provided
financial information and answer the question with precise numerical calculations.
"""

TEACHER_MODEL_ID = "llama3.2:1b"

client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    ),
    mode=instructor.Mode.JSON
)

def format_table(table_data):
    if not table_data or len(table_data) == 0:
        return "No table data provided."
    
    formatted_table = []
    for row in table_data:
        if isinstance(row, list):
            formatted_table.append(" | ".join(str(cell) for cell in row))
        else:
            formatted_table.append(str(row))
    
    return "\n".join(formatted_table)

def format_text_sequences(text_seq):
    if not text_seq:
        return ""
    if isinstance(text_seq, list):
        return "\n".join(text_seq)
    return str(text_seq)

def predict_answer(finqa_data, debug=False):
    """Generate zero-shot prediction for a FinQA sample."""
    if debug:
        print("\n" + "=" * 60)
        print(f"DEBUG INFO FOR SAMPLE ID: {finqa_data.get('id', 'unknown')}")
        print(f"Question: {finqa_data['question']}")
        print(f"Ground Truth: {finqa_data.get('final_result', 'N/A')}")
        print(f"Gold Instructions (gold_inds): {finqa_data.get('gold_inds', 'N/A')}")
        print(f"Program Reasoning (program_re): {finqa_data.get('program_re', 'N/A')}")
        print("=" * 60 + "\n")

    # Format the input data
    pre_text = format_text_sequences(finqa_data.get('pre_text', []))
    post_text = format_text_sequences(finqa_data.get('post_text', []))
    table = format_table(finqa_data.get('table', []))
    question = finqa_data['question']

    prompt = f"""You are a financial analyst. You must analyze the provided financial data and answer the question with precise calculations.

FINANCIAL DATA:
{pre_text}

{table}

{post_text}

QUESTION: {question}

INSTRUCTIONS:
1. Read the financial data carefully
2. Identify the specific numbers mentioned in the question
3. Find those exact numbers in the provided data
4. Perform the calculation step by step
5. CRITICAL: Your answer field must contain ONLY the final numerical value, no units, no extra text

ANSWER FORMAT EXAMPLES:
- For percentages: "15.2" (not "15.2%" or "15.2 percent")
- For millions: "41932" (not "$41932 million" or "41932 million")
- For ratios: "0.532" (not "53.2%" unless specifically asked for percentage)
- For negative values: "-32.1" (not "-$32.1 million")

CRITICAL: Extract only the core numerical value for the answer field. All explanations go in reasoning.

OUTPUT FORMAT (strict JSON):
```json
{{
"reasoning": "<show your step-by-step calculation here>",
"answer": "<ONLY the final numerical value, no units or formatting>"
}}
``` """

    try:
        completion = client.chat.completions.create(
            model=TEACHER_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_model=FinQAResponse,
            temperature=0.1,
            max_completion_tokens=512,
            top_p=0.9,
        )

        # Clean formatting for gold_inds and program_re
        gold_inds = finqa_data.get('gold_inds', [])
        if isinstance(gold_inds, str):
            try:
                gold_inds = json.loads(gold_inds)
            except Exception:
                gold_inds = []
        if not isinstance(gold_inds, list):
            gold_inds = [gold_inds]
        gold_inds = [int(i) for i in gold_inds if isinstance(i, (int, float, str)) and str(i).isdigit()]

        program_re = finqa_data.get('program_re', "")
        if not isinstance(program_re, str):
            program_re = str(program_re)

        return {
            "answer": completion.answer,
            "reasoning": completion.reasoning,
            "gold_inds": gold_inds,
            "program_re": program_re,
            "ground_truth": finqa_data.get('final_result', None)
        }

    except Exception as e:
        print(f"\nError generating answer: {str(e)}\n")
        # Try fallback with simpler prompt
        try:
            print("Trying fallback without structured output...\n")
            fallback_prompt = f"""Based on this financial data, answer the question with just a number:

{pre_text}
{table}
{post_text}

Question: {question}

Answer with only the numerical value:"""

            fallback_completion = client.chat.completions.create(
                model=TEACHER_MODEL_ID,
                messages=[
                    {
                        "role": "user",
                        "content": fallback_prompt
                    }
                ],
                temperature=0.1,
                max_completion_tokens=50,
                top_p=0.9,
            )

            # Extract answer from response
            answer = fallback_completion.choices[0].message.content.strip()
            
            # Clean formatting for gold_inds and program_re
            gold_inds = finqa_data.get('gold_inds', [])
            if isinstance(gold_inds, str):
                try:
                    gold_inds = json.loads(gold_inds)
                except Exception:
                    gold_inds = []
            if not isinstance(gold_inds, list):
                gold_inds = [gold_inds]
            gold_inds = [int(i) for i in gold_inds if isinstance(i, (int, float, str)) and str(i).isdigit()]

            program_re = finqa_data.get('program_re', "")
            if not isinstance(program_re, str):
                program_re = str(program_re)

            return {
                "answer": answer,
                "reasoning": "Fallback prediction",
                "gold_inds": gold_inds,
                "program_re": program_re,
                "ground_truth": finqa_data.get('final_result', None)
            }

        except Exception as e2:
            print(f"Fallback also failed: {str(e2)}\n")
            return {
                "answer": None,
                "reasoning": None,
                "gold_inds": [],
                "program_re": "",
                "ground_truth": finqa_data.get('final_result', None)
            }

def process_finqa_sample(sample_data, debug=False):
    """Process a single FinQA sample and return results."""
    pred = predict_answer(sample_data, debug=debug)

    if pred["answer"] is None:
        return {
            'id': sample_data.get('id', 'unknown'),
            'question': sample_data['question'],
            'ground_truth': pred["ground_truth"],
            'predicted_answer': None,
            'reasoning': pred["reasoning"],
            'gold_inds': pred["gold_inds"],
            'program_re': pred["program_re"],
            'success': False,
            'is_parsable': False
        }

    # Check if answer is parsable
    is_parsable = is_answer_parsable(pred["answer"])

    return {
        'id': sample_data.get('id', 'unknown'),
        'question': sample_data['question'],
        'ground_truth': pred["ground_truth"],
        'predicted_answer': pred["answer"],
        'reasoning': pred["reasoning"],
        'gold_inds': pred["gold_inds"],
        'program_re': pred["program_re"],
        'success': True,
        'is_parsable': is_parsable
    }

def calculate_current_metrics(results):
    """Calculate current accuracy and invalid answer count"""
    total_processed = len(results)
    valid_answers = [r for r in results if r['success'] and r['is_parsable']]
    invalid_answers = [r for r in results if not r['success'] or not r['is_parsable']]
    
    if len(valid_answers) == 0:
        return 0.0, len(invalid_answers), total_processed
    
    correct = 0
    for result in valid_answers:
        if are_answers_equivalent(result['predicted_answer'], result['ground_truth']):
            correct += 1
    
    accuracy = correct / len(valid_answers)
    return accuracy, len(invalid_answers), total_processed

def write_to_csv(results, filename="finqa_results.csv"):
    """Write results to CSV file"""
    fieldnames = ['id', 'question', 'ground_truth', 'predicted_answer', 'reasoning', 'gold_inds', 'program_re', 'success', 'is_parsable']
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            # Convert lists to string for CSV
            csv_result = result.copy()
            csv_result['gold_inds'] = json.dumps(result['gold_inds']) if result['gold_inds'] else "[]"
            
            # Clean up text fields to avoid CSV issues
            for field in ['question', 'reasoning']:
                if csv_result[field]:
                    csv_result[field] = str(csv_result[field]).replace('"', '""').replace('\n', ' ')
            
            writer.writerow(csv_result)

def write_metrics_to_csv(metrics, filename="finqa_metrics_log.csv"):
    fieldnames = [
        "accuracy", "successful", "invalid_incorrect", "correct"
    ]
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        # Only keep the relevant fields from metrics
        filtered_metrics = {k: metrics[k] for k in fieldnames if k in metrics}
        writer.writerow(filtered_metrics)

def write_simple_metrics_to_csv(steps, successful, unresolved, correct, invalid_incorrect, filename="student_metrics.csv"):
    fieldnames = [
        "Steps", "Successful samples", "Unresolved samples", "Correct samples", "Invalid/Incorrect cases"
    ]
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerow([
            steps,
            successful,
            unresolved,
            correct,
            invalid_incorrect
        ])

def create_combined_dataset(train_data, val_data, total_samples=1000):
    """Create a combined dataset with samples from both train and validation"""
    # Calculate how many samples to take from each split
    train_samples = min(500, len(train_data))  # Take up to 500 from train
    val_samples = min(total_samples - train_samples, len(val_data))  # Fill remaining from validation
    
    # If validation doesn't have enough, take more from train
    if val_samples < (total_samples - train_samples):
        remaining_needed = total_samples - train_samples - val_samples
        train_samples = min(train_samples + remaining_needed, len(train_data))
    
    print(f"Taking {train_samples} samples from train and {val_samples} samples from validation")
    
    # Randomly select samples
    train_indices = random.sample(range(len(train_data)), train_samples)
    val_indices = random.sample(range(len(val_data)), val_samples)
    
    # Combine samples with source information
    combined_samples = []
    for idx in train_indices:
        sample = train_data[idx]
        sample_dict = dict(sample)
        sample_dict['source'] = 'train'
        sample_dict['original_index'] = idx
        combined_samples.append(sample_dict)
    
    for idx in val_indices:
        sample = val_data[idx]
        sample_dict = dict(sample)
        sample_dict['source'] = 'validation'
        sample_dict['original_index'] = idx
        combined_samples.append(sample_dict)
    
    # Shuffle the combined samples
    random.shuffle(combined_samples)
    
    return combined_samples

def run_1000_sample_evaluation(train_data, val_data, max_workers=10):
    """Run evaluation on 1000 samples from both train and validation splits."""
    print(f"\n=== Running evaluation on 1000 samples from both train and validation ===\n")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create combined dataset
    combined_data = create_combined_dataset(train_data, val_data, 1000)
    print(f"Created combined dataset with {len(combined_data)} samples")
    
    results = []
    csv_filename = "finqa_1000_samples_results.csv"
    
    print(f"Starting evaluation with {max_workers} workers...")
    print("=" * 80)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {executor.submit(process_finqa_sample, sample, debug=(i < 2)): i
                           for i, sample in enumerate(combined_data)}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_sample)):
            try:
                result = future.result()
                results.append(result)
                
                # Live logging every 5 samples
                if (i + 1) % 5 == 0 or (i + 1) == len(combined_data):
                    current_accuracy, invalid_count, total_processed = calculate_current_metrics(results)
                    successful = sum(1 for r in results if r['success'] and r['is_parsable'])
                    unresolved = sum(1 for r in results if not r['success'] or not r['is_parsable'])
                    correct = sum(1 for r in results if r['success'] and r['is_parsable'] and are_answers_equivalent(r['predicted_answer'], r['ground_truth']))
                    invalid_incorrect = successful - correct + unresolved
                    # Write current results to CSV
                    write_to_csv(results, csv_filename)
                    # Log metrics to metrics CSV
                    metrics_row = {
                        "step": i + 1,
                        "accuracy": f"{current_accuracy*100:.2f}",
                        "successful": successful,
                        "invalid_incorrect": invalid_incorrect,
                        "correct": correct
                    }
                    write_metrics_to_csv(metrics_row)
                    print(f"\n--- Progress Update: {i + 1}/{len(combined_data)} samples processed ---")
                    print(f"Current Accuracy: {current_accuracy:.4f} ({current_accuracy*100:.2f}%)")
                    print(f"Successful samples: {successful} ({successful/total_processed*100:.2f}%)")
                    print(f"Unresolved samples: {unresolved} ({unresolved/total_processed*100:.2f}%)")
                    # Show last 5 sample IDs
                    recent_samples = results[-5:]
                    sample_ids = [r['id'] for r in recent_samples]
                    print(f"Recent Sample IDs: {sample_ids}")
                    # Show details of the most recent sample
                    if results:
                        last_result = results[-1]
                        print(f"Last Sample Details:")
                        print(f"  ID: {last_result['id']}")
                        print(f"  Question: {last_result['question'][:100]}...")
                        print(f"  Ground Truth: {last_result['ground_truth']}")
                        print(f"  Predicted: {last_result['predicted_answer']}")
                        print(f"  Parsable: {last_result['is_parsable']}")
                        if last_result['is_parsable'] and last_result['ground_truth']:
                            is_correct = are_answers_equivalent(last_result['predicted_answer'], last_result['ground_truth'])
                            print(f"  Correct: {is_correct}")
                    print("-" * 80)
                    
            except Exception as e:
                print(f"\nError processing sample {i}: {str(e)}\n")
                # Add a failed result
                results.append({
                    'id': f'failed_{i}',
                    'question': 'Failed to process',
                    'ground_truth': None,
                    'predicted_answer': None,
                    'reasoning': None,
                    'gold_inds': [],
                    'program_re': '',
                    'success': False,
                    'is_parsable': False
                })
    
    # Final CSV write
    write_to_csv(results, csv_filename)
    
    # Save results to JSON file as well
    results_file = "finqa_1000_samples_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate final metrics
    final_accuracy, final_invalid_count, total_processed = calculate_current_metrics(results)
    valid_count = total_processed - final_invalid_count
    
    print(f"\n" + "=" * 80)
    print(f"FINAL EVALUATION RESULTS")
    print(f"=" * 80)
    print(f"Total samples processed: {total_processed}")
    print(f"Valid answers: {valid_count}")
    print(f"Invalid/unparsable answers: {final_invalid_count}")
    print(f"Final accuracy (on valid answers): {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"Success rate: {(total_processed - final_invalid_count)/total_processed*100:.2f}%")
    print(f"Results saved to CSV: {csv_filename}")
    print(f"Results saved to JSON: {results_file}")
    print(f"=" * 80)
    
    # Show some examples of invalid answers
    invalid_results = [r for r in results if not r['success'] or not r['is_parsable']]
    if invalid_results:
        print(f"\nExamples of invalid answers:")
        for i, result in enumerate(invalid_results[:5]):
            print(f"  {i+1}. ID: {result['id']}, Answer: {result['predicted_answer']}")
    
    return results, final_accuracy

if __name__ == "__main__":
    print("Running evaluation on 1000 samples from both train and validation splits...")
    results, accuracy = run_1000_sample_evaluation(train_data, val_data, max_workers=10)
    print(f"\nEvaluation completed!")
    print(f"Final accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")