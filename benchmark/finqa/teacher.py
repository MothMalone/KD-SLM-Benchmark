from openai import OpenAI
import datasets
import dotenv
import os
import warnings
import concurrent.futures
import instructor
from pydantic import BaseModel, Field
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import pandas as pd
import json
import re
from typing import List, Dict, Any

warnings.filterwarnings("ignore")

class FinQAResponse(BaseModel):
    reasoning: str = Field(
        description="Step‑by‑step chain‑of‑thought showing how the calculation was done."
    )
    answer: str = Field(
        description="The final numerical or text answer, no extra formatting."
    )

    class Config:
        pass


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

TEACHER_MODEL_ID = "llama2:13b"

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
        print(f"Pre-text length: {len(format_text_sequences(finqa_data.get('pre_text', [])))} chars")
        print(f"Table length: {len(format_table(finqa_data.get('table', [])))} chars")
        print(f"Post-text length: {len(format_text_sequences(finqa_data.get('post_text', [])))} chars")
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

EXAMPLES:
- If asked for a percentage: answer like "15.2%" or "0.3%"
- If asked for millions: answer like "41932" or "1500 million"
- If asked for growth rate: calculate (new-old)/old * 100 and add %
- If asked for a ratio: divide the numbers and multiply by 100 for percentage

CRITICAL: Your answer must be derived from the actual data provided above. Do not guess or hallucinate numbers.

OUTPUT FORMAT (strict JSON):
```json
{{
  "reasoning": "<show your step-by-step calculation here>",
  "answer": "<just the final value>"
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
            temperature=0.1,  # Lower temperature for more consistent numerical answers
            max_completion_tokens=512,  # Reduced since we only want the answer
            top_p=0.9,
        )

        print("\n=== Model Reasoning ===")
        print(completion.reasoning)
        print(f"\nGenerated answer: {completion.answer}\n")

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
        # Try fallback without structured output
        try:
            print("Trying fallback without structured output...\n")
            fallback_completion = client.chat.completions.create(
                model=TEACHER_MODEL_ID,
                messages=[
                    {
                        "role": "user",
                        "content": prompt + "\n\nProvide only the final answer value:"
                    }
                ],
                response_model=FinQAResponse,
                temperature=0.1,
                max_completion_tokens=100,
                top_p=0.9,
            )

            answer = fallback_completion.choices[0].message.content.strip()
            # Clean formatting for gold_inds and program_re (same as above)
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
                "reasoning": None,
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
        print(f"Failed to generate answer for sample {sample_data.get('id', 'unknown')}")
        return {
            'id': sample_data.get('id', 'unknown'),
            'question': sample_data['question'],
            'ground_truth': pred["ground_truth"],
            'predicted_answer': None,
            'reasoning': pred["reasoning"],
            'gold_inds': pred["gold_inds"],
            'program_re': pred["program_re"],
            'success': False
        }

    return {
        'id': sample_data.get('id', 'unknown'),
        'question': sample_data['question'],
        'ground_truth': pred["ground_truth"],
        'predicted_answer': pred["answer"],
        'reasoning': pred["reasoning"],
        'gold_inds': pred["gold_inds"],
        'program_re': pred["program_re"],
        'success': True
    }

def run_evaluation(data, split_name, max_samples=None, max_workers=10):
    """Run evaluation on a dataset split."""
    print(f"\n=== Running evaluation on {split_name} split ===\n")
    
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))
        print(f"Using subset of {len(data)} samples for testing")
    
    results = []
    successful_predictions = 0
    failed_predictions = 0
    
    print(f"Starting evaluation with {max_workers} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {executor.submit(process_finqa_sample, sample, debug=(i < 3)): i
                           for i, sample in enumerate(data)}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_sample)):
            try:
                result = future.result()
                results.append(result)
                
                if result['success']:
                    successful_predictions += 1
                else:
                    failed_predictions += 1
                
                # Log progress every 10 samples
                if (i + 1) % 10 == 0 or (i + 1) == len(data):
                    print(f"\nProcessed {i + 1}/{len(data)} samples. "
                          f"Success: {successful_predictions}, Failed: {failed_predictions}\n")
                
                # Log sample details every 5 samples for debugging
                if (i + 1) % 5 == 0:
                    print(f"Sample {i + 1} - ID: {result['id']}")
                    print(f"  Question: {result['question'][:100]}...")
                    print(f"  Ground Truth: {result['ground_truth']}")
                    print(f"  Predicted: {result['predicted_answer']}")
                    print(f"  Success: {result['success']}")
                    print("-" * 50 + "\n")
                    
            except Exception as e:
                print(f"\nError processing sample {i}: {str(e)}\n")
                failed_predictions += 1
    
    # Save results to file
    results_file = f"finqa_{split_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation complete!")
    print(f"Total samples: {len(results)}")
    print(f"Successful predictions: {successful_predictions}")
    print(f"Failed predictions: {failed_predictions}")
    print(f"Success rate: {successful_predictions/len(results)*100:.2f}%")
    print(f"Results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    import sys

    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        print("Running full evaluation...")
        train_results = run_evaluation(train_data, "train", max_workers=10)
        val_results = run_evaluation(val_data, "validation", max_workers=10)
    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running test evaluation on 20 samples...") 
        test_results = run_evaluation(train_data, "train_test", max_samples=20, max_workers=5)
    else:
        # Default: Test on small subset first
        print("Starting with small test on 20 samples...")
        print("Use --test for 20 samples, --full for complete evaluation")
        test_results = run_evaluation(train_data, "train_test", max_samples=20, max_workers=5)

        print("\nSmall test completed. To run full evaluation, use:")
        print("python teacher_finqa.py --full")

