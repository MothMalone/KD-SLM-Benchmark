"""
Comprehensive FinQA evaluation script that runs teacher model, normalizes answers, and calculates metrics.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import subprocess
import pandas as pd
from finqa_answer_normalizer import FinQAAnswerNormalizer

def run_teacher_evaluation(max_samples=None, max_workers=10):
    """Run the teacher model evaluation."""
    print("=== Running Teacher Model Evaluation ===")
    
    # Import and run teacher evaluation
    try:
        import teacher
        
        # Run on train data first
        print("Running evaluation on training data...")
        train_results = teacher.run_evaluation(
            teacher.train_data, 
            "train", 
            max_samples=max_samples, 
            max_workers=max_workers
        )
        
        # Run on validation data
        print("Running evaluation on validation data...")
        val_results = teacher.run_evaluation(
            teacher.val_data, 
            "validation", 
            max_samples=max_samples, 
            max_workers=max_workers
        )
        
        return train_results, val_results
        
    except Exception as e:
        print(f"Error running teacher evaluation: {e}")
        return None, None

def normalize_and_evaluate(results_file, output_prefix=""):
    """Normalize answers and calculate evaluation metrics."""
    print(f"\n=== Normalizing and Evaluating {results_file} ===")
    
    normalizer = FinQAAnswerNormalizer()
    
    base_name = Path(results_file).stem
    output_file = f"{output_prefix}{base_name}_normalized.csv"
    
    # Normalize results
    df = normalizer.normalize_results(results_file, output_file)
    
    # Calculate detailed metrics
    total_samples = len(df)
    correct_samples = df['is_correct'].sum()
    accuracy = correct_samples / total_samples if total_samples > 0 else 0
    
    # Analyze error types
    error_df = df[~df['is_correct']]
    
    print(f"\n=== Detailed Results for {results_file} ===")
    print(f"Total samples: {total_samples}")
    print(f"Correct answers: {correct_samples}")
    print(f"Incorrect answers: {len(error_df)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Show some error examples
    if len(error_df) > 0:
        print(f"\n=== Error Analysis (showing first 3 errors) ===")
        for i, (_, row) in enumerate(error_df.head(3).iterrows()):
            print(f"Error {i+1}:")
            print(f"  ID: {row['id']}")
            print(f"  Question: {row['question'][:150]}...")
            print(f"  Ground Truth: {row['ground_truth']}")
            print(f"  Predicted: {row['predicted_answer']}")
            print(f"  Explanation: {row['evaluation_explanation']}")
            print()
    
    return df, accuracy

def generate_summary_report(train_df, val_df, output_file="finqa_evaluation_summary.txt"):
    """Generate a comprehensive summary report."""
    print(f"\n=== Generating Summary Report ===")
    
    with open(output_file, 'w') as f:
        f.write("FinQA Teacher Model Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Training results
        if train_df is not None:
            train_accuracy = train_df['is_correct'].mean()
            f.write(f"Training Set Results:\n")
            f.write(f"  Total samples: {len(train_df)}\n")
            f.write(f"  Correct answers: {train_df['is_correct'].sum()}\n")
            f.write(f"  Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)\n\n")
        
        # Validation results
        if val_df is not None:
            val_accuracy = val_df['is_correct'].mean()
            f.write(f"Validation Set Results:\n")
            f.write(f"  Total samples: {len(val_df)}\n")
            f.write(f"  Correct answers: {val_df['is_correct'].sum()}\n")
            f.write(f"  Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)\n\n")
        
        # Error analysis
        if train_df is not None:
            train_errors = train_df[~train_df['is_correct']]
            f.write(f"Training Set Error Examples:\n")
            for i, (_, row) in enumerate(train_errors.head(5).iterrows()):
                f.write(f"  Error {i+1}:\n")
                f.write(f"    ID: {row['id']}\n")
                f.write(f"    Question: {row['question'][:100]}...\n")
                f.write(f"    Ground Truth: {row['ground_truth']}\n")
                f.write(f"    Predicted: {row['predicted_answer']}\n")
                f.write(f"    Explanation: {row['evaluation_explanation']}\n\n")
        
        if val_df is not None:
            val_errors = val_df[~val_df['is_correct']]
            f.write(f"Validation Set Error Examples:\n")
            for i, (_, row) in enumerate(val_errors.head(5).iterrows()):
                f.write(f"  Error {i+1}:\n")
                f.write(f"    ID: {row['id']}\n")
                f.write(f"    Question: {row['question'][:100]}...\n")
                f.write(f"    Ground Truth: {row['ground_truth']}\n")
                f.write(f"    Predicted: {row['predicted_answer']}\n")
                f.write(f"    Explanation: {row['evaluation_explanation']}\n\n")
    
    print(f"Summary report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run FinQA teacher model evaluation")
    parser.add_argument("--max-samples", type=int, default=None, 
                       help="Maximum number of samples to evaluate (default: all)")
    parser.add_argument("--max-workers", type=int, default=10,
                       help="Maximum number of worker threads (default: 10)")
    parser.add_argument("--test-only", action="store_true",
                       help="Run on small test set only (20 samples)")
    parser.add_argument("--normalize-only", type=str, default=None,
                       help="Only normalize existing results file (provide path)")
    
    args = parser.parse_args()
    
    if args.test_only:
        args.max_samples = 20
        args.max_workers = 5
        print("Running in test mode with 20 samples and 5 workers")
    
    train_df = None
    val_df = None
    
    if args.normalize_only:
        # Only normalize existing results
        print(f"Normalizing existing results: {args.normalize_only}")
        train_df, _ = normalize_and_evaluate(args.normalize_only)
    else:
        # Run full evaluation pipeline
        print("Starting full evaluation pipeline...")
        
        # Run teacher model evaluation
        train_results, val_results = run_teacher_evaluation(
            max_samples=args.max_samples,
            max_workers=args.max_workers
        )
        
        if train_results is None:
            print("Teacher evaluation failed. Exiting.")
            return
        
        # Normalize and evaluate results
        if os.path.exists("finqa_train_results.json"):
            train_df, train_acc = normalize_and_evaluate("finqa_train_results.json", "train_")
        
        if os.path.exists("finqa_validation_results.json"):
            val_df, val_acc = normalize_and_evaluate("finqa_validation_results.json", "val_")
        
        # Generate summary report
        generate_summary_report(train_df, val_df)
    
    print("\n=== Evaluation Complete ===")
    if train_df is not None:
        print(f"Training accuracy: {train_df['is_correct'].mean():.4f}")
    if val_df is not None:
        print(f"Validation accuracy: {val_df['is_correct'].mean():.4f}")

if __name__ == "__main__":
    main()
