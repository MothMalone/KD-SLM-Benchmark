#!/usr/bin/env python3
"""
ROUGE evaluation script for DialogSum dataset comparing Llama 3.2 3B vs 1B models.
Tests both models on 500 samples from the test set using ROUGE metrics.
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
    def __init__(self, teacher_model_name="meta-llama/Llama-3.2-3B-Instruct", 
                 student_model_name="meta-llama/Llama-3.2-1B-Instruct",
                 device="auto"):
        """
        Initialize the evaluator with teacher and student models.
        
        Args:
            teacher_model_name: HuggingFace model name for teacher (3B)
            student_model_name: HuggingFace model name for student (1B)
            device: Device to run models on
        """
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        
        # Setup device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load models and tokenizers
        self.load_models()
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], 
            use_stemmer=True
        )
        
    def load_models(self):
        """Load teacher and student models with their tokenizers."""
        print("Loading teacher model (3B)...")
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(self.teacher_model_name)
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            self.teacher_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" if self.device == "cuda" else None
        )
        
        print("Loading student model (1B)...")
        self.student_tokenizer = AutoTokenizer.from_pretrained(self.student_model_name)
        self.student_model = AutoModelForCausalLM.from_pretrained(
            self.student_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Set pad tokens if they don't exist
        if self.teacher_tokenizer.pad_token is None:
            self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
        if self.student_tokenizer.pad_token is None:
            self.student_tokenizer.pad_token = self.student_tokenizer.eos_token
            
        print("Models loaded successfully!")
    
    def create_prompt(self, dialogue):
        """Create a prompt for dialogue summarization."""
        prompt = f"""Please summarize the following dialogue concisely:

Dialogue:
{dialogue}

Summary:"""
        return prompt
    
    def generate_summary(self, model, tokenizer, dialogue, max_new_tokens=128):
        """Generate summary using the specified model."""
        prompt = self.create_prompt(dialogue)
        
        # Tokenize input
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,  
                temperature=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the summary part (after "Summary:")
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
        
        # Calculate averages
        avg_scores = {}
        for metric in scores.keys():
            avg_scores[metric] = {
                'mean': np.mean(scores[metric]),
                'std': np.std(scores[metric]),
                'scores': scores[metric]
            }
            
        return avg_scores
    
    def write_metrics_to_csv(self, teacher_scores, student_scores, current_sample, filename="dialogsum_metrics.csv"):
        """Write current metrics to CSV file."""
        import csv
        import os
        
        # Define headers for new file
        fieldnames = ['samples', 'metric', 'teacher_mean', 'teacher_std', 'student_mean', 'student_std', 'gap']
        
        # Check if file exists
        file_exists = os.path.isfile(filename)
        
        with open(filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Write header if new file
            if not file_exists:
                writer.writeheader()
            
            # Write current metrics
            for metric in teacher_scores.keys():
                row = {
                    'samples': current_sample,
                    'metric': metric,
                    'teacher_mean': f"{teacher_scores[metric]['mean']:.4f}",
                    'teacher_std': f"{teacher_scores[metric]['std']:.4f}",
                    'student_mean': f"{student_scores[metric]['mean']:.4f}",
                    'student_std': f"{student_scores[metric]['std']:.4f}",
                    'gap': f"{teacher_scores[metric]['mean'] - student_scores[metric]['mean']:.4f}"
                }
                writer.writerow(row)

    def evaluate_on_dialogsum(self, num_samples=500, save_results=True, seed=42):
        """Evaluate both models on DialogSum dataset."""
        print("Loading DialogSum dataset...")
        dataset = load_dataset("knkarthick/dialogsum", split="test")
        
        # Sample the specified number of examples
        if num_samples > len(dataset):
            num_samples = len(dataset)
            print(f"Dataset has only {len(dataset)} samples, using all of them.")
        np.random.seed(seed)
        # Randomly sample examples
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        sampled_dataset = dataset.select(indices)
        
        print(f"Evaluating on {num_samples} samples...")
        
        # Store results
        results = {
            'teacher_summaries': [],
            'student_summaries': [],
            'reference_summaries': [],
            'dialogues': [],
            'teacher_scores': None,
            'student_scores': None
        }
        
        # Process examples in batches of 10
        for i, example in enumerate(tqdm(sampled_dataset, desc="Processing")):
            dialogue = example['dialogue']
            reference = example['summary']
            
            # Generate teacher summary
            teacher_summary = self.generate_summary(
                self.teacher_model, self.teacher_tokenizer, dialogue
            )
            
            # Generate student summary
            student_summary = self.generate_summary(
                self.student_model, self.student_tokenizer, dialogue
            )
            
            # Store results
            results['teacher_summaries'].append(teacher_summary)
            results['student_summaries'].append(student_summary)
            results['reference_summaries'].append(reference)
            results['dialogues'].append(dialogue)
            
            # Every 10 samples, compute and log intermediate metrics
            if (i + 1) % 10 == 0:
                print(f"\nComputing metrics after {i + 1} samples...")
                current_teacher_scores = self.compute_rouge_scores(
                    results['teacher_summaries'], results['reference_summaries']
                )
                current_student_scores = self.compute_rouge_scores(
                    results['student_summaries'], results['reference_summaries']
                )
                
                # Log to CSV
                self.write_metrics_to_csv(
                    current_teacher_scores, 
                    current_student_scores, 
                    i + 1
                )
        
        # Compute final ROUGE scores
        print("\nComputing final ROUGE scores...")
        results['teacher_scores'] = self.compute_rouge_scores(
            results['teacher_summaries'], results['reference_summaries']
        )
        results['student_scores'] = self.compute_rouge_scores(
            results['student_summaries'], results['reference_summaries']
        )
        
        # Print results
        self.print_results(results)
        
        # Save final results
        if save_results:
            self.save_results(results, num_samples)
            
        return results
    
    def print_results(self, results):
        """Print evaluation results in a formatted way."""
        print("\n" + "="*60)
        print("ROUGE EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nTeacher Model ({self.teacher_model_name}):")
        print("-" * 50)
        for metric, scores in results['teacher_scores'].items():
            print(f"{metric.upper()}: {scores['mean']:.4f} (±{scores['std']:.4f})")
        
        print(f"\nStudent Model ({self.student_model_name}):")
        print("-" * 50)
        for metric, scores in results['student_scores'].items():
            print(f"{metric.upper()}: {scores['mean']:.4f} (±{scores['std']:.4f})")
        
        print(f"\nPerformance Gap (Teacher - Student):")
        print("-" * 50)
        for metric in results['teacher_scores'].keys():
            gap = results['teacher_scores'][metric]['mean'] - results['student_scores'][metric]['mean']
            print(f"{metric.upper()}: {gap:.4f}")
        
        print("\nSample Outputs:")
        print("-" * 50)
        for i in range(min(3, len(results['dialogues']))):
            print(f"\nExample {i+1}:")
            print(f"Dialogue: {results['dialogues'][i][:200]}...")
            print(f"Reference: {results['reference_summaries'][i]}")
            print(f"Teacher: {results['teacher_summaries'][i]}")
            print(f"Student: {results['student_summaries'][i]}")
    
    def save_results(self, results, num_samples):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rouge_evaluation_{num_samples}samples_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if key.endswith('_scores'):
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
    parser = argparse.ArgumentParser(description="Evaluate Llama models on DialogSum using ROUGE")
    parser.add_argument("--teacher-model", default="meta-llama/Llama-3.2-3B-Instruct", 
                       help="Teacher model name")
    parser.add_argument("--student-model", default="meta-llama/Llama-3.2-1B-Instruct", 
                       help="Student model name")
    parser.add_argument("--num-samples", type=int, default=500, 
                       help="Number of samples to evaluate")
    parser.add_argument("--device", default="auto", 
                       help="Device to run on (auto, cuda, cpu)")
    parser.add_argument("--no-save", action="store_true", 
                       help="Don't save results to file")
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = DialogSumEvaluator(
        teacher_model_name=args.teacher_model,
        student_model_name=args.student_model,
        device=args.device
    )
    
    # Run evaluation
    results = evaluator.evaluate_on_dialogsum(
        num_samples=args.num_samples,
        save_results=not args.no_save,
        seed=args.seed
    )
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()