import re
import string
import secrets
import evaluate
from collections import Counter
import torch

def _normalize(s):
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def _f1_score_sentence(prediction, answer):
    prediction_tokens = prediction.split()
    answer_tokens = answer.split()
    
    common = Counter(prediction_tokens) & Counter(answer_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0, 0, 0
    
    precision = num_common / len(prediction_tokens)
    recall = num_common / len(answer_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall

def f1_score(predictions, answers):
    f1_scores, precision_scores, recall_scores = [], [], []

    for prediction, answer_list in zip(predictions, answers):
        prediction = _normalize(prediction)
        max_f1, max_precision, max_recall = 0, 0, 0

        if not answer_list:
          if prediction == "" or 'no response' in prediction:
            max_f1 = max_precision = max_recall = 1
          else:
            max_f1 = max_precision = max_recall = 0
        else:
          if isinstance(answer_list, str): answer_list = [answer_list]
          for answer in answer_list:
            answer = _normalize(answer)
            f1, precision, recall = _f1_score_sentence(prediction, answer)
            max_f1, max_precision, max_recall = max(f1, max_f1), max(precision, max_precision), max(recall, max_recall)

        f1_scores.append(max_f1)
        precision_scores.append(max_precision)
        recall_scores.append(max_recall)

    average_f1 = sum(f1_scores) / len(f1_scores)
    average_precision = sum(precision_scores) / len(precision_scores)
    average_recall = sum(recall_scores) / len(recall_scores)

    return {'f1': average_f1, 'precision': average_precision, 'recall': average_recall}

def exact_match(predictions, answers):
    exact_match_scores = []
    for prediction, answer_list in zip(predictions, answers):
        prediction = _normalize(prediction)
        if isinstance(answer_list, str): answer_list = [answer_list]
        answer_list = [_normalize(item) for item in answer_list]
        if not answer_list and prediction == "" or "no response" in prediction: exact_match_scores.append(1)
        if prediction in answer_list: exact_match_scores.append(1)
        else: exact_match_scores.append(0)
    return sum(exact_match_scores)/len(exact_match_scores)

def rouge(predictions, answers):
    rouge_metric = evaluate.load('rouge', experiment_id=f"{secrets.randbelow(10000)}")
    if True:
      return rouge_metric.compute(predictions=predictions, references=answers)
    else:
      max_score = 0
      average = 0
       # Iterate over predictions and corresponding sets of reference answers
      for prediction, references in zip(predictions, answers):
        scores = []
        for reference in references:
            # Compute the ROUGE score for each reference
            score = rouge_metric.compute(predictions=[prediction], references=[reference])
            scores.append(score)
        average += (scores[0]["rougeLsum"]+scores[1]["rougeLsum"]+scores[2]["rougeLsum"])/3
        max_score +=  max(score["rougeLsum"] for score in scores)
        # print(rouge_metric.compute(predictions=[prediction], references=[references]))
        # print(max_score)
        # input()
        
      return {"rougeLsum":average/len(predictions)}

def bert_score1(predictions, answers):
    bertscore = evaluate.load("bertscore", experiment_id=f"{secrets.randbelow(10000)}")
    if isinstance(answers[0], dict):
      f1, precision, recall = [0]*len(predictions), [0]*len(predictions), [0]*len(predictions)
      for i, row in enumerate(answers):
        for answer in row:
          tmp = bertscore.compute(predictions=predictions[i], references=answer, lang="en", rescale_with_baseline=True)
          f1[i] = max(f1[i], tmp['f1'])
          precision[i] = max(precision[i], tmp['precision'])
          recall[i] = max(recall[i], tmp['recall'])
      return {'f1': f1, 'precision': precision, 'recall': recall}
    else:
      return bertscore.compute(predictions=predictions, references=answers, lang="en", rescale_with_baseline=True)
    
    
def bert_score(predictions, answers, device='cpu'):
    # 设置PyTorch默认使用的设备为CPU
    device = torch.device(device)
    bertscore = evaluate.load("bertscore", experiment_id=f"{secrets.randbelow(10000)}", device=str(device))
    
    f1, precision, recall = [0] * len(predictions), [0] * len(predictions), [0] * len(predictions)

    # 如果answers的第一个元素是字典，则假设每个answers元素都是字典
    if isinstance(answers[0], dict):
        for i, row in enumerate(answers):
            for answer in row.values():  # 确保从字典中正确地提取答案
                tmp = bertscore.compute(predictions=[predictions[i]], references=[answer], lang="en", rescale_with_baseline=True, device=str(device))
                print(f"Debug: pred={predictions[i]}, answer={answer}, scores={tmp}")
                # 确保使用列表中的第一个元素进行比较和更新
                f1[i] = max(f1[i], tmp['f1'][0])
                precision[i] = max(precision[i], tmp['precision'][0])
                recall[i] = max(recall[i], tmp['recall'][0])
    else:
        # 否则假设answers是直接的答案列表
        for i, answer in enumerate(answers):
            tmp = bertscore.compute(predictions=[predictions[i]], references=[answer], lang="en", rescale_with_baseline=True, device=str(device))
            print(f"Debug: pred={predictions[i]}, answer={answer}, scores={tmp}")
            # 确保使用列表中的第一个元素进行比较和更新
            f1[i] = max(f1[i], tmp['f1'][0])
            precision[i] = max(precision[i], tmp['precision'][0])
            recall[i] = max(recall[i], tmp['recall'][0])

    # 返回计算得到的f1, precision, recall列表
    return {'f1': f1, 'precision': precision, 'recall': recall}



