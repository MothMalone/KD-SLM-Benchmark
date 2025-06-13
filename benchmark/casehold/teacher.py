from openai import OpenAI
import datasets
import dotenv
import os
import warnings
import concurrent.futures
import instructor
from pydantic import BaseModel, Field
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

class CaseholdAnswer(BaseModel):
    answer: int = Field(
        None,
        description="The index of the correct holding statement, which can be 0, 1, 2, 3, or 4.",
        ge=0,
        le=4
    )

dotenv.load_dotenv('../../.env')
ds = datasets.load_dataset("MothMalone/SLMS-KD-Benchmarks", "casehold")

data = ds['train']
print(len(data), "cases loaded from the dataset.")
# data = data.select(range(10))
DATASET_CONTEXT = """
CaseHOLD is a multiple choice question answering task derived
from legal citations in judicial rulings. The citing context from the
judicial decision serves as the prompt for the question. The answer
choices are holding statements derived from citations following
text in a legal decision. The correct answer is the holding statement that corresponds
to the citing text. The four incorrect answers are other holding
statements.
"""

TEACHER_MODEL_ID = "llama3.1:8b"

client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    ),
    mode=instructor.Mode.JSON
)

def predict_answer(case_data):
    completion = client.chat.completions.create(
        model=TEACHER_MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": f"""
                {DATASET_CONTEXT}
                Given the following data from the casehold dataset: 
                {case_data['citing_prompt']} 
                with holdings:
                {case_data['holding_0']}, 
                {case_data['holding_1']}, 
                {case_data['holding_2']}, 
                {case_data['holding_3']}, 
                {case_data['holding_4']} 

                Please provide the index of the correct holding statement, which must be a number between 0 and 4, inclusive.
                Only output the number (0, 1, 2, 3, or 4).
                """
            }
        ],
        response_model=CaseholdAnswer,
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
    )
    print(f"Predicted answer for case: {completion.answer}")
    return completion.answer

def process_case(case_data):
    predicted_answer = predict_answer(case_data)
    
    if predicted_answer not in {0, 1, 2, 3, 4}:
        print(f"Invalid answer received: {predicted_answer}. Skipping this case.")
        return case_data['label'], predicted_answer
    actual_answer = case_data['label']
    return actual_answer, predicted_answer

y_true = []
y_pred = []
invalid_cases = 0

metrics_df = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1 Score", "Invalid Cases"])

with concurrent.futures.ThreadPoolExecutor(max_workers= 100) as executor:
    results = executor.map(process_case, data)

    for i, (actual_answer, predicted_answer) in enumerate(results):
        if predicted_answer is not None:
            y_true.append(actual_answer)
            y_pred.append(predicted_answer)
        else:
            invalid_cases += 1

        if (i + 1) % 100 == 0 or (i + 1) == len(data):
            y_true_array = np.array(y_true, dtype=float)
            y_pred_array = np.array(y_pred, dtype=float)

            accuracy = accuracy_score(y_true_array, y_pred_array)
            precision = precision_score(y_true_array, y_pred_array, average='weighted')
            recall = recall_score(y_true_array, y_pred_array, average='weighted')
            f1 = f1_score(y_true_array, y_pred_array, average='weighted')
            conf_matrix = confusion_matrix(y_true_array, y_pred_array)

            metrics_row = pd.DataFrame([{
                "Accuracy": accuracy * 100,
                "Precision": precision * 100,
                "Recall": recall * 100,
                "F1 Score": f1 * 100,   
                "Invalid Cases": invalid_cases
            }])

            metrics_df = pd.concat([metrics_df, metrics_row], ignore_index=True)

            metrics_df.to_csv("metrics_log.csv", index=False, mode='a')
            print(f"Logged metrics after {i + 1} cases.")

y_true_array = np.array(y_true, dtype=float)
y_pred_array = np.array(y_pred, dtype=float)

accuracy = accuracy_score(y_true_array, y_pred_array)
precision = precision_score(y_true_array, y_pred_array, average='weighted')
recall = recall_score(y_true_array, y_pred_array, average='weighted')
f1 = f1_score(y_true_array, y_pred_array, average='weighted')
conf_matrix = confusion_matrix(y_true_array, y_pred_array)

print(f"Final Accuracy: {accuracy * 100:.2f}%")
print(f"Final Precision: {precision * 100:.2f}%")
print(f"Final Recall: {recall * 100:.2f}%")
print(f"Final F1 Score: {f1 * 100:.2f}%")
print("Confusion Matrix:")
print(f"Invalid Cases: {invalid_cases}")
print(conf_matrix)