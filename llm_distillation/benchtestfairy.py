import os
import json
import logging
import argparse
from itertools import chain
from tqdm import tqdm
from score import f1_score, exact_match, rouge, bert_score

def load_predictions(file_path):
    predictions = []
    answers = []
    with open(file_path, 'r') as file:
        for line in file:
            item = json.loads(line)
            predictions.append(item['prediction_text'][0])
            answers.append(item['answers'])
    return predictions, answers

def compute_scores(predictions, answers, args):
    logging.info('Computing scores...')
    results = f1_score(predictions, answers)
    results['em'] = exact_match(predictions, answers)
    results['squad'] = (results['f1'] + results['em']) / 2
    results.update(rouge(predictions, answers))
    if args.bert_score:
        results.update(bert_score(predictions, answers))
    

    for key, value in results.items():
        if isinstance(value, list):  
            if value:  

                if all(isinstance(v, (int, float)) for v in value):
                    average_value = sum(value) / len(value)  
                    results[key] = round(average_value * 100, 2)  
                else:
                    results[key] = 'Invalid data'  
            else:
                results[key] = 0  
        elif isinstance(value, (int, float)):  
            results[key] = round(value * 100, 2)  
        else:
            results[key] = 'Invalid data'  

    return results


def save_results(output, args, results):
    titled_folder = "titled" if args.title else "untitled"
    output = args.output_path if args.output_path else f"{os.getenv('HOME')}/Multi-Level-OT/llm_distillation/benchmark/results/{args.model_id.split('/')[-1]}/{args.dataset_id.split('/')[-1]}/{titled_folder}"
    os.makedirs(output, exist_ok=True)
    
    with open(f"{output}/results.json", 'w') as json_file:
        json.dump(results, json_file, indent=4)
    logging.info("Results saved.")

def main(args):
    logging.basicConfig(level=logging.INFO)
    logging.info('Start processing...')

    predictions, answers = load_predictions(args.prediction_file)

    results = compute_scores(predictions, answers, args)
    logging.info(f"Computed Results: {results}")

    save_results(args.output_path, args, results)
    logging.info("Process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to compute scores from predictions.")
    parser.add_argument("--model_id", type=str, default="model123", help="Model ID")
    parser.add_argument("--dataset_id", type=str, help="Dataset ID")
    parser.add_argument("--prediction_file", type=str, help="File path for predictions and answers")
    parser.add_argument("--title", action="store_true", help="To keep title in the results")
    parser.add_argument("--bert_score", action="store_true", help="To compute bert score")
    parser.add_argument("--output_path", type=str, default="", help="Output path")
    args = parser.parse_args()

    main(args)
