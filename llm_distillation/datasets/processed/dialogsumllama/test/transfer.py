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
    

    arrow_df['summary_llama'] = None
    arrow_df['answers'] = None


    json_index = 0
    arrow_index = 0
    while json_index < len(json_df) and arrow_index < len(arrow_df):
        prediction_text = json_df.at[json_index, 'prediction_text']
        for _ in range(3):  
            if arrow_index < len(arrow_df):
                arrow_df.at[arrow_index, 'summary_llama'] = prediction_text
                arrow_df.at[arrow_index, 'answers'] = json_df.at[json_index, 'answers'][0]
                arrow_index += 1
        json_index += 1

    return pa.Table.from_pandas(arrow_df)

def main(arrow_file_path, json_file_path, output_arrow_file_path):
    arrow_table = load_arrow_file(arrow_file_path)
    json_df = load_json_file(json_file_path)
    
    updated_table = update_and_clean_arrow_table(arrow_table, json_df)
    
    with pa.OSFile(output_arrow_file_path, 'wb') as sink:
        writer = pa.ipc.new_stream(sink, updated_table.schema)
        writer.write_table(updated_table)
        writer.close()

    print("Updated Arrow file has been saved.")


main('data-00000-of-00001.arrow', 'predictions_3shots.json', 'newdata.arrow')
