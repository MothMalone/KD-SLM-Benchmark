import datasets
import dotenv
from teacher import predict_answer, format_table, format_text_sequences


dotenv.load_dotenv('../../.env')
ds = datasets.load_dataset("MothMalone/SLMS-KD-Benchmarks", "finqa")
train_data = ds['train']

def test_single_sample(index=0):
    sample = train_data[index]
    
    print("=" * 80)
    print(f"TESTING SAMPLE {index}: {sample['id']}")
    print("=" * 80)
    
    print(f"Question: {sample['question']}")
    print(f"Ground Truth: {sample['final_result']}")
    print()
    
    # Show the raw data
    print("RAW DATA:")
    print(f"Pre-text: {sample.get('pre_text', [])}")
    print(f"Table: {sample.get('table', [])}")
    print(f"Post-text: {sample.get('post_text', [])}")
    print()
    
    # Show formatted data
    pre_text = format_text_sequences(sample.get('pre_text', []))
    post_text = format_text_sequences(sample.get('post_text', []))
    table = format_table(sample.get('table', []))
    
    print("FORMATTED DATA:")
    print("Pre-text:")
    print(pre_text[:1000] + "..." if len(pre_text) > 1000 else pre_text)
    print()
    print("Table:")
    print(table[:1000] + "..." if len(table) > 1000 else table)
    print()
    print("Post-text:")
    print(post_text[:1000] + "..." if len(post_text) > 1000 else post_text)
    print()
    
    # Test prediction
    print("RUNNING PREDICTION...")
    predicted_answer, reasoning = predict_answer(sample, debug=True)
    
    print(f"Predicted Answer: {predicted_answer}")
    print(f"Ground Truth: {sample['final_result']}")
    if reasoning:
        print("\nModel reasoning:\n")
        print(reasoning)
    else:
        print("\nNo reasoning returned.")

    # Ground truth programs
    print("\n=== GOLD INSTRUCTIONS DEBUG ===")
    print("Gold Instructions (gold_ins):")
    print(sample.get('gold_inds', 'N/A'))
    print()
    print("Program Reasoning (program_re):")
    print(sample.get('program_re', 'N/A'))
    print()

    
   

def test_multiple_samples():
    for i in range(45,47):
        test_single_sample(i)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    test_multiple_samples() 
    #test_single_sample(1)
    
   
