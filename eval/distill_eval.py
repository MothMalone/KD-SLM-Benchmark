#!/usr/bin/env python3
"""
ROUGE evaluation script for DialogSum dataset for a single model.
Tests the model on N samples from the test set using ROUGE metrics.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from rouge_score import rouge_scorer
import json
import argparse
from tqdm import tqdm
import os
from datetime import datetime

class DialogSumEvaluator:
    def __init__(self, model_name, device="auto"):
        """
        Initialize the evaluator with a single model.
        Args:
            model_name: HuggingFace model name
            device: Device to run model on
        """
        self.model_name = model_name
        # Setup device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"Using device: {self.device}")
        # Load model and tokenizer
        self.load_model()
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], 
            use_stemmer=True
        )

    def load_model(self):
        """Load model and tokenizer."""
        print(f"Loading model: {self.model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Model loaded successfully!")

    def create_prompt(self, dialogue):
        """Create a prompt for dialogue summarization."""
        prompt = f"""Please summarize the following dialogue concisely:\n\nDialogue:\n{dialogue}\n\nSummary:"""
        return prompt

    def generate_summary(self, dialogue, max_new_tokens=128):
        """Generate summary using the model."""
        prompt = self.create_prompt(dialogue)
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048,
            padding=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,  
                temperature=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Summary:" in generated_text:
            summary = generated_text.split("Summary:")[-1].strip()
        else:
            summary = generated_text[len(prompt):].strip()
        return summary

    def compute_rouge_scores(self, predictions, references):
        """Compute ROUGE scores for predictions vs references."""
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': []}
        for pred, ref in zip(predictions, references):
            rouge_score = self.rouge_scorer.score(ref, pred)
            for metric in scores.keys():
                scores[metric].append(rouge_score[metric].fmeasure)
        avg_scores = {}
        for metric in scores.keys():
            avg_scores[metric] = {
                'mean': np.mean(scores[metric]),
                'std': np.std(scores[metric]),
                'scores': scores[metric]
            }
        return avg_scores

    def write_metrics_to_csv(self, scores, current_sample, filename=None):
        """Write current metrics to CSV file."""
        import csv
        if filename is None:
            model_id = self.model_name.replace('/', '_')
            filename = f"dialogsum_metrics_{model_id}.csv"
        fieldnames = ['samples', 'metric', 'mean', 'std']
        file_exists = os.path.isfile(filename)
        with open(filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for metric in scores.keys():
                row = {
                    'samples': current_sample,
                    'metric': metric,
                    'mean': f"{scores[metric]['mean']:.4f}",
                    'std': f"{scores[metric]['std']:.4f}"
                }
                writer.writerow(row)

    def evaluate_on_dialogsum(self, num_samples=500, save_results=True, seed=42):
        """Evaluate the model on DialogSum dataset."""
        print("Loading DialogSum dataset...")
        dataset = load_dataset("knkarthick/dialogsum", split="test")
        if num_samples > len(dataset):
            num_samples = len(dataset)
            print(f"Dataset has only {len(dataset)} samples, using all of them.")
        np.random.seed(seed)
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        sampled_dataset = dataset.select(indices)
        print(f"Evaluating on {num_samples} samples...")
        results = {
            'summaries': [],
            'reference_summaries': [],
            'dialogues': [],
            'scores': None
        }
        for i, example in enumerate(tqdm(sampled_dataset, desc="Processing")):
            dialogue = example['dialogue']
            reference = example['summary']
            summary = self.generate_summary(dialogue)
            results['summaries'].append(summary)
            results['reference_summaries'].append(reference)
            results['dialogues'].append(dialogue)
            if (i + 1) % 10 == 0:
                print(f"\nComputing metrics after {i + 1} samples...")
                current_scores = self.compute_rouge_scores(
                    results['summaries'], results['reference_summaries']
                )
                self.write_metrics_to_csv(
                    current_scores, 
                    i + 1
                )
        print("\nComputing final ROUGE scores...")
        results['scores'] = self.compute_rouge_scores(
            results['summaries'], results['reference_summaries']
        )
        self.print_results(results)
        if save_results:
            self.save_results(results, num_samples)
        return results

    def print_results(self, results):
        """Print evaluation results in a formatted way."""
        print("\n" + "="*60)
        print("ROUGE EVALUATION RESULTS")
        print("="*60)
        print(f"\nModel ({self.model_name}):")
        print("-" * 50)
        for metric, scores in results['scores'].items():
            print(f"{metric.upper()}: {scores['mean']:.4f} (±{scores['std']:.4f})")
        print("\nSample Outputs:")
        print("-" * 50)
        for i in range(min(3, len(results['dialogues']))):
            print(f"\nExample {i+1}:")
            print(f"Dialogue: {results['dialogues'][i][:200]}...")
            print(f"Reference: {results['reference_summaries'][i]}")
            print(f"Model: {results['summaries'][i]}")

    def save_results(self, results, num_samples):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = self.model_name.replace('/', '_')
        filename = f"rouge_evaluation_{model_id}_{num_samples}samples_{timestamp}.json"
        json_results = {}
        for key, value in results.items():
            if key == 'scores':
                json_results[key] = {}
                for metric, scores in value.items():
                    json_results[key][metric] = {
                        'mean': float(scores['mean']),
                        'std': float(scores['std']),
                        'scores': [float(s) for s in scores['scores']]
                    }
            else:
                json_results[key] = value
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate one or more models on DialogSum using ROUGE")
    parser.add_argument("--model", help="Model name (single model, for backward compatibility)")
    parser.add_argument("--models", nargs='+', help="List of model names to evaluate (space-separated)")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of samples to evaluate")
    parser.add_argument("--device", default="auto", help="Device to run on (auto, cuda, cpu)")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Determine models to evaluate
    if args.models:
        model_names = args.models
    elif args.model:
        model_names = [args.model]
    else:
        raise ValueError("You must specify either --model or --models.")

    for model_name in model_names:
        print(f"\n{'='*80}\nEvaluating model: {model_name}\n{'='*80}")
        evaluator = DialogSumEvaluator(
            model_name=model_name,
            device=args.device
        )
        results = evaluator.evaluate_on_dialogsum(
            num_samples=args.num_samples,
            save_results=not args.no_save,
            seed=args.seed
        )
        print("\nEvaluation complete for:", model_name)

if __name__ == "__main__":
    main()