import datasets
import json
from finqa_answer_normalizer import FinQAAnswerNormalizer

def test_dataset_loading():
    """Test that we can load the FinQA dataset correctly."""
    print("=== Testing Dataset Loading ===")
    
    try:
        ds = datasets.load_dataset("MothMalone/SLMS-KD-Benchmarks", "finqa")
        
        print(f"Dataset loaded successfully!")
        print(f"Splits available: {list(ds.keys())}")
        
        for split_name, split_data in ds.items():
            print(f"{split_name}: {len(split_data)} samples")
        
        # Check first sample structure
        sample = ds['train'][0]
        print(f"\nSample structure:")
        for key, value in sample.items():
            if isinstance(value, list):
                print(f"  {key}: list with {len(value)} items")
                if len(value) > 0:
                    print(f"    First item: {str(value[0])[:100]}...")
            else:
                print(f"  {key}: {str(value)[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False

def test_answer_normalizer():
    print("\n=== Testing Answer Normalizer ===")
    
    normalizer = FinQAAnswerNormalizer()
    
    # Test cases based on the evaluation rules
    test_cases = [
        ("0.98", "98%", True),  # Format difference but same value
        ("2", "1.98", True),    # Rounding difference
        ("100", "100.0", True), # Decimal vs integer
        ("50%", "0.5", True),   # Percentage conversion
        ("1000", "1,000", True), # Comma formatting
        ("5", "10", False),     # Different values
        ("yes", "no", False),   # Different text
        ("2.5 million", "2500000", True), # Scale conversion
    ]
    
    correct_count = 0
    for i, (ground_truth, model_answer, expected) in enumerate(test_cases):
        is_correct, explanation = normalizer.evaluate_answer_pair(ground_truth, model_answer)
        
        print(f"Test {i+1}: GT='{ground_truth}' vs Model='{model_answer}'")
        print(f"  Expected: {expected}, Got: {is_correct}")
        print(f"  Explanation: {explanation}")
        
        if is_correct == expected:
            correct_count += 1
            print("  ✓ PASS")
        else:
            print("  ✗ FAIL")
        print()
    
    print(f"Normalizer test results: {correct_count}/{len(test_cases)} passed")
    return correct_count == len(test_cases)

def create_sample_results():
    """Create sample results file for testing normalization."""
    print("\n=== Creating Sample Results File ===")
    
    sample_results = [
        {
            "id": "test_1",
            "question": "What is 50% of 200?",
            "ground_truth": "100",
            "predicted_answer": "100.0",
            "reasoning": "50% of 200 = 0.5 * 200 = 100",
            "success": True
        },
        {
            "id": "test_2", 
            "question": "What percentage is 25 out of 100?",
            "ground_truth": "25%",
            "predicted_answer": "0.25",
            "reasoning": "25/100 = 0.25 = 25%",
            "success": True
        },
        {
            "id": "test_3",
            "question": "Round 1.98 to nearest integer",
            "ground_truth": "2",
            "predicted_answer": "1.98",
            "reasoning": "1.98 rounds to 2",
            "success": True
        },
        {
            "id": "test_4",
            "question": "What is 1000 + 500?",
            "ground_truth": "1500",
            "predicted_answer": "The answer is 1,500",
            "reasoning": "1000 + 500 = 1500",
            "success": True
        },
        {
            "id": "test_5",
            "question": "Failed prediction test",
            "ground_truth": "42",
            "predicted_answer": None,
            "reasoning": None,
            "success": False
        }
    ]
    
    with open("sample_finqa_results.json", "w") as f:
        json.dump(sample_results, f, indent=2)
    
    print("Sample results file created: sample_finqa_results.json")
    return "sample_finqa_results.json"

def test_normalization_pipeline():
    """Test the complete normalization pipeline."""
    print("\n=== Testing Normalization Pipeline ===")
    
    # Create sample results
    results_file = create_sample_results()
    
    # Test normalization
    normalizer = FinQAAnswerNormalizer()
    df = normalizer.normalize_results(results_file, "sample_normalized_results.csv")
    
    print(f"\nNormalization completed!")
    print(f"Results shape: {df.shape}")
    print(f"Accuracy: {df['is_correct'].mean():.4f}")
    
    # Show results
    print("\nSample results:")
    for i, row in df.iterrows():
        print(f"  {row['id']}: {row['ground_truth']} vs {row['predicted_answer']} -> {row['is_correct']}")
    
    return True

def main():
    print("FinQA Evaluation Setup Test")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Dataset loading
    if test_dataset_loading():
        tests_passed += 1
        print("✓ Dataset loading test PASSED")
    else:
        print("✗ Dataset loading test FAILED")
    
    # Test 2: Answer normalizer
    if test_answer_normalizer():
        tests_passed += 1
        print("✓ Answer normalizer test PASSED")
    else:
        print("✗ Answer normalizer test FAILED")
    
    # Test 3: Sample results creation
    try:
        create_sample_results()
        tests_passed += 1
        print("✓ Sample results creation PASSED")
    except Exception as e:
        print(f"✗ Sample results creation FAILED: {e}")
    
    # Test 4: Normalization pipeline
    try:
        test_normalization_pipeline()
        tests_passed += 1
        print("✓ Normalization pipeline test PASSED")
    except Exception as e:
        print(f"✗ Normalization pipeline test FAILED: {e}")
    
    print(f"\n=== Test Summary ===")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("All tests passed! Setup is ready.")
        print("\nNext steps:")
        print("1. Run: python teacher_finqa.py --test")
        print("2. Run: python finqa_evaluator.py --test-only")
        print("3. If tests work, run full evaluation with --full")
    else:
        print("Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
