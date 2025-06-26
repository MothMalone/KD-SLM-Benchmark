import instructor
import datasets
import warnings
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

class CaseholdAnswer(BaseModel):
    answer: int = Field(..., description="The index (0-4) of the correct holding.", ge=0, le=4)

ds = datasets.load_dataset("MothMalone/SLMS-KD-Benchmarks", "casehold")
data = ds['train'].select(range(1000))
print(f"{len(data)} cases loaded for evaluation.")

DATASET_CONTEXT = """
CaseHOLD is a multiple choice question answering task derived from legal citations in judicial rulings. The citing context from the judicial decision serves as the prompt for the question. The answer choices are holding statements derived from citations following text in a legal decision. The correct answer is the holding statement that corresponds to the citing text. The four incorrect answers are other holding statements.
"""


STUDENT_MODEL_ID = "MothMalone/Llama3.2-1B-Casehold-Distilled"  

student_model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_ID, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

client = instructor.from_transformers(
    student_model,
    tokenizer,
    mode=instructor.Mode.JSON,
)


def predict_answer(case_data):
    """Generates a structured prediction for a given case."""
    try:
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    {DATASET_CONTEXT}
                    Given the following data from the casehold dataset: 
                    {case_data['citing_prompt']} 
                    with holdings:
                    0: {case_data['holding_0']}
                    1: {case_data['holding_1']}
                    2: {case_data['holding_2']}
                    3: {case_data['holding_3']}
                    4: {case_data['holding_4']} 

                    Please provide the index of the correct holding statement, which must be a number between 0 and 4.
                    """
                }
            ],
            response_model=CaseholdAnswer,
            max_retries=2,
        )
        return completion.answer
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

y_true = []
y_pred = []
invalid_cases = 0
metrics_df = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1 Score", "Macro F1 Score", "Invalid Cases"])

for i, case_data in enumerate(tqdm(data, desc="Evaluating Student Model")):
    predicted_answer = predict_answer(case_data)
    
    if predicted_answer is not None and predicted_answer in {0, 1, 2, 3, 4}:
        y_true.append(int(case_data['label']))
        y_pred.append(predicted_answer)
    else:
        invalid_cases += 1
        print(f"Invalid answer received: {predicted_answer}. Skipping.")

    # Log metrics periodically
    if (i + 1) % 10 == 0 or (i + 1) == len(data):
        if not y_pred: continue # Skip if no valid predictions yet
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        new_row = pd.DataFrame([{"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1, "Macro F1 Score": f1_macro, "Invalid Cases": invalid_cases}])
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
        metrics_df.to_csv("metrics_log_student.csv", index=False)

print("\n--- Final Student Model Inference Results ---")
if len(y_pred) > 0:
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(5)))

    print(f"Final Accuracy: {accuracy * 100:.2f}%")
    print(f"Final Precision (Weighted): {precision * 100:.2f}%")
    print(f"Final Recall (Weighted): {recall * 100:.2f}%")
    print(f"Final F1 Score (Weighted): {f1 * 100:.2f}%")
    print(f"Final Macro F1 Score: {f1_macro * 100:.2f}%")
    print(f"Total Invalid/Failed Cases: {invalid_cases}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
else:
    print("No valid predictions were made to calculate final metrics.")
