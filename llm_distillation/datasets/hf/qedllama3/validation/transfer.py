import pandas as pd
import pyarrow as pa
import json
from tqdm import tqdm

def load_json_file(json_file):
    with open(json_file, 'r') as file:
        data = [json.loads(line) for line in file]
    return pd.DataFrame(data)

def load_arrow_file(file_path):
    with pa.memory_map(file_path, 'r') as source:
        table = pa.ipc.open_stream(source).read_all()
    return table

def update_and_clean_arrow_table(arrow_table, json_df):
    arrow_df = arrow_table.to_pandas()
    original_answers = []
    matched_indices = set()  # 用于存储匹配的行索引

    # 初始化列以确保它们存在
    arrow_df['llama3_8b_chat_answers'] = None
    arrow_df['answers'] = None

    # 遍历每一行
    for index, row in tqdm(arrow_df.iterrows(), total=len(arrow_df), desc="Processing rows"):
        original_answer = row['original_nq_answers'][:][0]['string']

        if original_answer and original_answer in json_df['answers'].values:
            matched_indices.add(index)
            answer_index = json_df[json_df['answers'] == original_answer].index[0]
            predicted_text = json_df.at[answer_index, 'prediction_text']
            if isinstance(predicted_text, list):
                predicted_text = predicted_text[0] if predicted_text else None
            arrow_df.at[index, 'llama3_8b_chat_answers'] = predicted_text
            arrow_df.at[index, 'answers'] = json_df.at[answer_index, 'answers']
        else:
            original_answers.append(original_answer)

    # 删除未匹配的行
    all_indices = set(arrow_df.index)
    unmatched_indices = list(all_indices - matched_indices)
    arrow_df.drop(unmatched_indices, inplace=True)
    arrow_df.reset_index(drop=True, inplace=True)

    return pa.Table.from_pandas(arrow_df), len(unmatched_indices)

def main(arrow_file_path, json_file_path, output_arrow_file_path):
    arrow_table = load_arrow_file(arrow_file_path)
    json_df = load_json_file(json_file_path)
    
    updated_table, num_deleted = update_and_clean_arrow_table(arrow_table, json_df)
    
    with pa.OSFile(output_arrow_file_path, 'wb') as sink:
        writer = pa.ipc.new_stream(sink, updated_table.schema)
        writer.write_table(updated_table)
        writer.close()

    print("Updated Arrow file has been saved.")
    print(f"Deleted {num_deleted} entries due to no match found.")

# 调用主函数
main('data-00000-of-00001.arrow', 'predictions_5shots.json', 'newdata.arrow')
