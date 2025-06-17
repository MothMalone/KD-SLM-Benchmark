import re
import json
import numpy as np
from typing import Union, Optional, Tuple
import pandas as pd

class FinQAAnswerNormalizer:
    """
    Normalizes FinQA model answers to match ground truth format according to evaluation rules.
    
    Rules:
    1. If ground truth is numerical and format differs but values are same, consider consistent
       (e.g., 0.98 vs 98%)
    2. If ground truth is numerical and model answer rounds to ground truth, consider consistent
       (e.g., ground truth 2 vs model answer 1.98)
    """
    
    def __init__(self, tolerance=0.01):
        self.tolerance = tolerance
        
    def extract_number(self, text: str) -> Optional[float]:
        if text is None:
            return None

        text = str(text).strip()

        # Only accept if the text is a single number (possibly with % or $ or scale word)
        # Reject if text contains arithmetic operators or is a sentence/expression
        if re.search(r'[+\-*/=]', text) or len(text.split()) > 3:
            return None

        # Handle percentage format (e.g., "98%", "98.5%")
        percentage_match = re.fullmatch(r'(-?\d+(?:\.\d+)?)\s*%', text)
        if percentage_match:
            return float(percentage_match.group(1)) / 100

        # Handle currency format (e.g., "$1,234.56", "1,234.56 million")
        currency_match = re.fullmatch(r'[\$]?([\d,]+(?:\.\d+)?)(?:\s*(million|billion|thousand|m|b|k))?', text, re.IGNORECASE)
        if currency_match:
            number_str = currency_match.group(1).replace(',', '')
            number = float(number_str)
            scale = currency_match.group(2)
            if scale:
                scale = scale.lower()
                if scale in ['million', 'm']:
                    number *= 1_000_000
                elif scale in ['billion', 'b']:
                    number *= 1_000_000_000
                elif scale in ['thousand', 'k']:
                    number *= 1_000
            return number

        # Handle simple decimal numbers
        decimal_match = re.fullmatch(r'-?\d+(?:\.\d+)?', text)
        if decimal_match:
            return float(decimal_match.group())

        return None
    
    def normalize_text(self, text: str) -> str:
        """Normalize text by removing extra whitespace and standardizing format."""
        if text is None:
            return ""
        return str(text).strip().lower()
    
    def are_numbers_equivalent(self, num1: float, num2: float, tolerance: float = None) -> bool:
        """Check if two numbers are equivalent within tolerance or rounding rules."""
        if tolerance is None:
            tolerance = self.tolerance
            
        # Direct equality check
        if abs(num1 - num2) < tolerance:
            return True
        
        # Check if one rounds to the other
        if abs(round(num1) - num2) < tolerance or abs(num1 - round(num2)) < tolerance:
            return True
        
        # Check percentage conversion (0.98 vs 98)
        if abs(num1 * 100 - num2) < tolerance or abs(num1 - num2 * 100) < tolerance:
            return True
        
        return False
    
    def evaluate_answer_pair(self, ground_truth: str, model_answer: str) -> Tuple[bool, str]:
        """
        Evaluate if model answer matches ground truth according to FinQA rules.
        
        Returns:
            Tuple[bool, str]: (is_match, explanation)
        """
        print(f"Evaluating: GT='{ground_truth}' vs Model='{model_answer}'")
        
        if ground_truth is None or model_answer is None:
            return False, "One or both answers are None"
        
        # Extract numbers from both answers
        gt_number = self.extract_number(ground_truth)
        model_number = self.extract_number(model_answer)
        
        print(f"  Extracted numbers: GT={gt_number}, Model={model_number}")
        
        # If both are numbers, apply numerical rules
        if gt_number is not None and model_number is not None:
            is_equivalent = self.are_numbers_equivalent(gt_number, model_number)
            explanation = f"Numerical comparison: {gt_number} vs {model_number}, equivalent={is_equivalent}"
            print(f"  {explanation}")
            return is_equivalent, explanation
        
        # If ground truth is number but model answer is not, try to extract from model
        if gt_number is not None and model_number is None:
            # Try more aggressive extraction
            model_text = str(model_answer).lower()
            
            # Look for "yes/no" answers that might correspond to binary questions
            if gt_number in [0, 1]:
                if any(word in model_text for word in ['no', 'false', 'zero', 'none']):
                    model_number = 0
                elif any(word in model_text for word in ['yes', 'true', 'one']):
                    model_number = 1
                    
                if model_number is not None:
                    is_equivalent = self.are_numbers_equivalent(gt_number, model_number)
                    explanation = f"Binary conversion: {gt_number} vs {model_number}, equivalent={is_equivalent}"
                    print(f"  {explanation}")
                    return is_equivalent, explanation
        
        # If neither is a clear number, do text comparison
        gt_normalized = self.normalize_text(ground_truth)
        model_normalized = self.normalize_text(model_answer)
        
        is_match = gt_normalized == model_normalized
        explanation = f"Text comparison: '{gt_normalized}' vs '{model_normalized}', match={is_match}"
        print(f"  {explanation}")
        
        return is_match, explanation
    
    def normalize_results(self, results_file: str, output_file: str = None) -> pd.DataFrame:
        """
        Normalize results from teacher evaluation and calculate metrics.
        
        Args:
            results_file: Path to JSON file with evaluation results
            output_file: Optional path to save normalized results
            
        Returns:
            DataFrame with normalized results and evaluation metrics
        """
        print(f"Loading results from {results_file}...")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print(f"Loaded {len(results)} results")
        
        normalized_results = []
        correct_count = 0
        total_evaluated = 0
        
        for i, result in enumerate(results):
            if not result['success']:
                print(f"Skipping failed prediction {i}: {result['id']}")
                continue
            
            ground_truth = result['ground_truth']
            predicted_answer = result['predicted_answer']
            
            is_correct, explanation = self.evaluate_answer_pair(ground_truth, predicted_answer)
            
            normalized_result = {
                'id': result['id'],
                'question': result['question'],
                'ground_truth': ground_truth,
                'predicted_answer': predicted_answer,
                'reasoning': result.get('reasoning', ''),
                'is_correct': is_correct,
                'evaluation_explanation': explanation,
                'sample_index': i
            }
            
            normalized_results.append(normalized_result)
            
            if is_correct:
                correct_count += 1
            total_evaluated += 1
            
            # Log every few samples for debugging
            if (i + 1) % 5 == 0:
                print(f"Sample {i + 1}:")
                print(f"  ID: {result['id']}")
                print(f"  Question: {result['question'][:100]}...")
                print(f"  Ground Truth: {ground_truth}")
                print(f"  Predicted: {predicted_answer}")
                print(f"  Correct: {is_correct}")
                print(f"  Explanation: {explanation}")
                print("-" * 60)
        
        # Calculate metrics
        accuracy = correct_count / total_evaluated if total_evaluated > 0 else 0
        
        print(f"\n=== Evaluation Results ===")
        print(f"Total samples evaluated: {total_evaluated}")
        print(f"Correct answers: {correct_count}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Create DataFrame
        df = pd.DataFrame(normalized_results)
        
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Normalized results saved to: {output_file}")
        
        return df

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python finqa_answer_normalizer.py <results_file.json> [output_file.csv]")
        return
    
    results_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    normalizer = FinQAAnswerNormalizer()
    df = normalizer.normalize_results(results_file, output_file)
    
    print("\n=== Sample Results ===")
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        print(f"Sample {i+1}:")
        print(f"  Ground Truth: {row['ground_truth']}")
        print(f"  Predicted: {row['predicted_answer']}")
        print(f"  Correct: {row['is_correct']}")
        print(f"  Explanation: {row['evaluation_explanation']}")
        print()

if __name__ == "__main__":
    main()
