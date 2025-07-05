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
import wandb
import csv

class DialogSumEvaluator:
    def __init__(self, model_name, device="auto"):
        """
        Initialize the evaluator with a single model for zero-shot evaluation.
        Args:
            model_name: HuggingFace model name
            device: Device to run model on
        """
        self.model_name = model_name
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"Using device: {self.device}")
        
        self.load_model()
        
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], 
            use_stemmer=True
        )

    def load_model(self):
        """Load model and tokenizer in evaluation mode only."""
        print(f"Loading model: {self.model_name} for zero-shot evaluation...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model in evaluation mode
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True  # For some models that require it
        )
        
        # Set model to evaluation mode (no training)
        self.model.eval()
        
        # Ensure no gradients are computed
        for param in self.model.parameters():
            param.requires_grad = False
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Model loaded successfully in evaluation mode!")

    def create_prompt(self, dialogue):
        prompt = f"Please summarize the following dialogue:\n\n{dialogue}\n\nSummary:"
        return prompt

    def generate_summary(self, dialogue, max_new_tokens=64):
        """Generate summary using the model with deterministic settings for zero-shot evaluation."""
        prompt = self.create_prompt(dialogue)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024,
            padding=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate with no gradient computation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  
                num_beams=1,      # Greedy decoding
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.0, 
                length_penalty=1.0       
            )
        

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Summary:" in generated_text:
            summary = generated_text.split("Summary:")[-1].strip()
        else:
            # Fallback: extract text after the prompt
            summary = generated_text[len(prompt):].strip()
        
        summary = " ".join(summary.split())
        
        if not summary:
            summary = "No summary generated."
        
        return summary

    def compute_rouge_scores(self, predictions, references):
        """Compute ROUGE scores for predictions vs references."""
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': []}
        
        for pred, ref in zip(predictions, references):
            pred_clean = " ".join(pred.split())
            ref_clean = " ".join(ref.split())
            
            if not pred_clean:
                pred_clean = "empty"
            
            rouge_score = self.rouge_scorer.score(ref_clean, pred_clean)
            
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

    def write_metrics_to_csv(self, scores, current_sample, run_name, filename=None):
        """Write current metrics to CSV file with specified format."""
        if filename is None:
            filename = f"dialogsum_rouge_results.csv"
        
        fieldnames = ['run', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        file_exists = os.path.isfile(filename)
        
        with open(filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            
            row = {
                'run': f"{run_name}_{current_sample}",
                'rouge1': f"{scores['rouge1']['mean']:.4f}",
                'rouge2': f"{scores['rouge2']['mean']:.4f}",
                'rougeL': f"{scores['rougeL']['mean']:.4f}",
                'rougeLsum': f"{scores['rougeLsum']['mean']:.4f}"
            }
            writer.writerow(row)

    def evaluate_on_dialogsum(self, num_samples=500, save_results=True, seed=42):
        """Evaluate the model on DialogSum dataset in zero-shot setting."""
        print("Loading DialogSum dataset...")
        dataset = load_dataset("knkarthick/dialogsum", split="test")
        
        if num_samples > len(dataset):
            num_samples = len(dataset)
            print(f"Dataset has only {len(dataset)} samples, using all of them.")
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        sampled_dataset = dataset.select(indices)
        
        print(f"Evaluating {self.model_name} on {num_samples} samples (zero-shot)...")
        
        run_name = f"{self.model_name.replace('/', '_')}_{num_samples}samples_zeroshot"
        wandb.init(
            project="dialogsum-rouge-evaluation",
            name=run_name,
            config={
                "model_name": self.model_name,
                "num_samples": num_samples,
                "seed": seed,
                "dataset": "dialogsum",
                "max_new_tokens": 64,
                "evaluation_type": "rouge",
                "setting": "zero_shot"
            }
        )
        
        results = {
            'summaries': [],
            'reference_summaries': [],
            'dialogues': [],
            'scores': None
        }
        
        # Process each example
        for i, example in enumerate(tqdm(sampled_dataset, desc="Zero-shot evaluation")):
            dialogue = example['dialogue']
            reference = example['summary']
            
            # Generate summary (zero-shot)
            try:
                summary = self.generate_summary(dialogue)
            except Exception as e:
                print(f"Error generating summary for sample {i}: {e}")
                summary = "Error in generation"
            
            results['summaries'].append(summary)
            results['reference_summaries'].append(reference)
            results['dialogues'].append(dialogue)
            
            if (i + 1) % 10 == 0:
                print(f"\nComputing metrics after {i + 1} samples...")
                current_scores = self.compute_rouge_scores(
                    results['summaries'], results['reference_summaries']
                )
                
                wandb.log({
                    "samples": i + 1,
                    "rouge1": current_scores['rouge1']['mean'],
                    "rouge2": current_scores['rouge2']['mean'],
                    "rougeL": current_scores['rougeL']['mean'],
                    "rougeLsum": current_scores['rougeLsum']['mean']
                })
                
           
                self.write_metrics_to_csv(current_scores, i + 1, run_name)
        
        # Compute final scores
        print("\nComputing final ROUGE scores...")
        results['scores'] = self.compute_rouge_scores(
            results['summaries'], results['reference_summaries']
        )
        
        wandb.log({
            "final_rouge1": results['scores']['rouge1']['mean'],
            "final_rouge2": results['scores']['rouge2']['mean'],
            "final_rougeL": results['scores']['rougeL']['mean'],
            "final_rougeLsum": results['scores']['rougeLsum']['mean']
        })
        
        self.print_results(results)
        
        if save_results:
            self.save_results(results, num_samples)
        
        wandb.finish()
        return results

    def print_results(self, results):
        """Print evaluation results in a formatted way."""
        print("\n" + "="*60)
        print("ZERO-SHOT ROUGE EVALUATION RESULTS")
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
            print(f"Generated: {results['summaries'][i]}")

    def save_results(self, results, num_samples):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = self.model_name.replace('/', '_')
        filename = f"zero_shot_rouge_evaluation_{model_id}_{num_samples}samples_{timestamp}.json"
        
        # Convert to JSON-serializable format
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
        
        json_results['metadata'] = {
            'model_name': self.model_name,
            'evaluation_type': 'zero_shot',
            'num_samples': num_samples,
            'timestamp': timestamp
        }
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate models on DialogSum using ROUGE (Zero-Shot)")
    parser.add_argument("--models", nargs='+', default=["meta-llama/Llama-3.2-1B","meta-llama/Llama-3.2-3B","bigscience/bloom-560m", "EleutherAI/pythia-410m", "facebook/opt-350m"], 
                       help="List of model names to evaluate")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of samples to evaluate")
    parser.add_argument("--device", default="auto", help="Device to run on (auto, cuda, cpu)")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()

    # Set wandb API key (replace with your own or use environment variable)
    if "WANDB_API_KEY" not in os.environ:
        os.environ["WANDB_API_KEY"] = "445d8df72343591d6588f101349ad4752497ce62"

    print("="*80)
    print("ZERO-SHOT DIALOGUE SUMMARIZATION EVALUATION")
    print("="*80)
    print(f"Models to evaluate: {args.models}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print("="*80)

    # Evaluate each model
    for model_name in args.models:
        print(f"\n{'='*80}\nEvaluating model: {model_name} (Zero-Shot)\n{'='*80}")
        
        try:
            evaluator = DialogSumEvaluator(
                model_name=model_name,
                device=args.device
            )
            
            results = evaluator.evaluate_on_dialogsum(
                num_samples=args.num_samples,
                save_results=not args.no_save,
                seed=args.seed
            )
            
            print(f"\nZero-shot evaluation complete for: {model_name}")
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            continue

if __name__ == "__main__":
    main()