from openai import OpenAI
import datasets
import dotenv
import os
import warnings
import instructor
from pydantic import BaseModel, Field, constr, ValidationError, field_validator
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
from typing import Literal

warnings.filterwarnings("ignore")

# Define the model for the answer
class PubmedqaAnswer(BaseModel):
    answer: Literal["yes", "no", "maybe"]

# Load environment variables
dotenv.load_dotenv('../../.env')

# Load dataset
ds = datasets.load_dataset("MothMalone/SLMS-KD-Benchmarks", "pubmedqa")
data = ds['train']
print(len(data), "cases loaded from the dataset.")

# Dataset context
DATASET_CONTEXT = """
PubMedQA is a dataset and a task that involves Question Answering (QA) using scientific literature from PubMed, which is a free resource that contains millions of articles related to life sciences and biomedical research. PubMedQA specifically focuses on using abstracts and passages from PubMed articles to answer medical and scientific questions.
"""

# Define the model ID
TEACHER_MODEL_ID = "llama3.2:1b"

# Initialize OpenAI instructor client
client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    ),
    mode=instructor.Mode.JSON
)

# Define a label encoding function
label_mapping = {
    "yes": 1,
    "no": 0,
    "maybe": 2
}

# Function to encode categorical labels to numerical values
def encode_labels(label):
    return label_mapping.get(label, -1)  # Returns -1 if label is invalid

# Predict function
def predict_answer(case_data):
    try:
        completion = client.chat.completions.create(
            model=TEACHER_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    {DATASET_CONTEXT}
                    Given the following question from the PubMedQA dataset: 
                    question : {case_data['question']}, 
                    with these data:
                    context : {case_data['context']}, 
                    long_answer {case_data['long_answer']}, 
                    Please provide the final decision based on these data and provide the final_decision. It is either
                    "yes" or "no" or "maybe".
                    """
                }
            ],
            response_model=PubmedqaAnswer,
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            max_retries=3,  
        )
        print(f"Predicted answer for ques: {completion.answer}")
        return completion.answer
    
    except ValidationError as e:
        print(e.errors())
        return None
    except Exception as e:
        print(e)
        return None

# Process case function
def process_case(case_data):
    predicted_answer = predict_answer(case_data)
    
    if predicted_answer not in {"yes", "no", "maybe"}:
        print(f"Invalid answer received: {predicted_answer}. Skipping this case.")
        return case_data['final_decision'], predicted_answer
    actual_answer = case_data['final_decision']
    return actual_answer, predicted_answer

# Initialize metrics and results
y_true = []
y_pred = []
invalid_cases = 0
metrics_df = pd.DataFrame(columns=["Accuracy", "Precision", "Recall", "F1 Score", "Macro F1 Score", "Invalid Cases"])

# Loop through dataset and process each case
for i, case_data in enumerate(data):
    actual_answer, predicted_answer = process_case(case_data)
    
    if predicted_answer is not None:
        # Encode both actual and predicted answers
        y_true.append(encode_labels(actual_answer))
        y_pred.append(encode_labels(predicted_answer))
    else:
        invalid_cases += 1

    # Calculate metrics periodically (every 10th case or at the end)
    if (i + 1) % 10 == 0 or (i + 1) == len(data):
        # Convert lists to numpy arrays for metric calculations
        y_true_array = np.array(y_true, dtype=int)
        y_pred_array = np.array(y_pred, dtype=int)

        # Calculate metrics
        accuracy = accuracy_score(y_true_array, y_pred_array)
        precision = precision_score(y_true_array, y_pred_array, average='weighted', zero_division=0)
        recall = recall_score(y_true_array, y_pred_array, average='weighted', zero_division=0)
        f1 = f1_score(y_true_array, y_pred_array, average='weighted', zero_division=0)
        f1_macro = f1_score(y_true_array, y_pred_array, average='macro', zero_division=0)
        conf_matrix = confusion_matrix(y_true_array, y_pred_array)

        # Log metrics to a DataFrame
        metrics_row = pd.DataFrame([{
            "Accuracy": round(accuracy * 100, 3),
            "Precision": round(precision * 100, 3),
            "Recall": round(recall * 100, 3),
            "F1 Score": round(f1 * 100, 3),
            "Macro F1 Score": round(f1_macro * 100, 3),   
            "Invalid Cases": invalid_cases
        }])
        metrics_df = pd.concat([metrics_df, metrics_row], ignore_index=True)

        # Save metrics to CSV
        metrics_df.to_csv("metrics_log.csv", index=False, mode='w')
        print(f"Logged metrics after {i + 1} cases.")

# Final metrics after processing all cases
y_true_array = np.array(y_true, dtype=int)
y_pred_array = np.array(y_pred, dtype=int)

accuracy = accuracy_score(y_true_array, y_pred_array)
precision = precision_score(y_true_array, y_pred_array, average='weighted', zero_division=0)
recall = recall_score(y_true_array, y_pred_array, average='weighted', zero_division=0)
f1 = f1_score(y_true_array, y_pred_array, average='weighted', zero_division=0)
f1_macro = f1_score(y_true_array, y_pred_array, average='macro', zero_division=0)
conf_matrix = confusion_matrix(y_true_array, y_pred_array)

# Print final results
print(f"Final Accuracy: {accuracy * 100:.2f}%")
print(f"Final Precision: {precision * 100:.2f}%")
print(f"Final Recall: {recall * 100:.2f}%")
print(f"Final F1 Score: {f1 * 100:.2f}%")
print(f"Final Macro F1 Score: {f1_macro * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print(f"Invalid Cases: {invalid_cases}")
