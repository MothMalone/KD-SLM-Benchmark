import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd
import numpy as np
import json
import nltk

# 确保已下载相关的nltk数据包
# nltk.download('punkt')

# 读取Arrow文件
file_path = 'data-00000-of-00001.arrow'
with open(file_path, 'rb') as f:
    reader = ipc.open_stream(f)
    table = reader.read_all()

# 将Arrow Table转换为Pandas DataFrame
df = table.to_pandas()

# 定义一个函数来计算词数
def word_count(text):
    words = nltk.word_tokenize(text)
    return len(words)

# 在DataFrame中新增一列计算question + paragraph_text的词数
df['word_count'] = df.apply(lambda row: word_count(row['question']) + word_count(row['paragraph_text']), axis=1)

# 按词数升序排列并取出前40行
df_sorted = df.nsmallest(80, 'word_count')

# 读取json文件
with open('predictions_0shots.json', 'r') as f:
    predictions = [json.loads(line) for line in f]

with open('predictions_0shots2.json', 'r') as f:
    predictions2 = [json.loads(line) for line in f]

# 定义一个函数来比较original_nq_answers和预测答案，并提取对应的两个prediction_text
def extract_prediction_texts(original_answers, predictions, predictions2):
    prediction_text = None
    distillation_text = None
    
    for sublist in original_answers:
        if isinstance(sublist, dict) and 'string' in sublist:
            answer = sublist['string']
            
            for pred in predictions:
                if answer in pred['prediction_text']:
                    prediction_text = pred['prediction_text']
                    break  # 假设只取第一个匹配项

            for pred2 in predictions2:
                if answer in pred2['prediction_text']:
                    distillation_text = pred2['prediction_text']
                    break  # 假设只取第一个匹配项
    
    return prediction_text, distillation_text

# 提取四十行中的original_nq_answers的string并与两个json文件中的prediction_text进行比较
df_sorted[['prediction_text', 'distillation_text']] = df_sorted['original_nq_answers'].apply(
    lambda x: pd.Series(extract_prediction_texts(x, predictions, predictions2))
)

# 只保留有匹配的行
df_filtered = df_sorted.dropna(subset=['prediction_text', 'distillation_text'])

# 转换DataFrame中的ndarray为list
df_filtered = df_filtered.applymap(lambda x: x.tolist() if isinstance(x, (list, np.ndarray)) else x)

# 将结果保存到新的json文件中
result = df_filtered[['paragraph_text', 'question', 'original_nq_answers', 'prediction_text', 'distillation_text']].to_dict(orient='records')

# 保存为新的JSON文件
with open('filtered_results_with_distillation.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print("处理完成，结果已保存到filtered_results_with_distillation.json文件中。")
