import pandas as pd
from datasets import Dataset, DatasetDict
import os
import json

# 设置路径
base_path = "/mnt/xieliang.xl/mo.zhu/workspace/LLMRecipes/llm-recipes/llm_distillation/datasets/hf"

# 读取 Parquet 文件
train_data_path = os.path.join(base_path, "data/train-00000-of-00001.parquet")
validation_data_path = os.path.join(base_path, "data/validation-00000-of-00001.parquet")

# 确保文件存在
if not os.path.exists(train_data_path) or not os.path.exists(validation_data_path):
    raise FileNotFoundError("One or both Parquet files are not found.")

train_data = pd.read_parquet(train_data_path)
validation_data = pd.read_parquet(validation_data_path)
print("train_data",train_data)

# 确保数据的 'answers' 列格式正确
# 假设您的原始数据需要这样的处理
train_data['answers'] = train_data['answers'].apply(lambda x: [{"end": ans['end'], "start": ans['start'], "string": ans['string']} for ans in x])
validation_data['answers'] = validation_data['answers'].apply(lambda x: [{"end": ans['end'], "start": ans['start'], "string": ans['string']} for ans in x])

# 转换为 Dataset 对象
train_dataset = Dataset.from_pandas(train_data)
validation_dataset = Dataset.from_pandas(validation_data)

# 保存路径
train_save_path = os.path.join(base_path, "qed/train")
validation_save_path = os.path.join(base_path, "qed/validation")

# 创建目录
os.makedirs(train_save_path, exist_ok=True)
os.makedirs(validation_save_path, exist_ok=True)

# 保存到 Arrow 格式
train_dataset.save_to_disk(train_save_path)
validation_dataset.save_to_disk(validation_save_path)

# 生成数据集信息
dataset_info = {
    "info": {
        "builder_name": "parquet_to_arrow",
        "build_time": "2024-05-24T12:00:00",
        "description": "QED dataset in Arrow format converted from Parquet",
        "license": "CC-BY-SA 3.0",
        "name": "qed_arrow",
        "version": "1.0.0"
    },
    "features": {
        "context": {"dtype": "string", "_type": "Value"},
        "question": {"dtype": "string", "_type": "Value"},
        "answers": {
            "_type": "Sequence",
            "feature": {
                "end": {"dtype": "int64", "_type": "Value"},
                "start": {"dtype": "int64", "_type": "Value"},
                "text": {"dtype": "string", "_type": "Value"}
            }
        },
        "answers_generated": {"dtype": "string", "_type": "Value"}
    },
    "citation": "",
    "homepage": "",
    "license": ""
}

# 保存数据集信息到文件
info_files = {
    "train": os.path.join(train_save_path, "dataset_info.json"),
    "validation": os.path.join(validation_save_path, "dataset_info.json")
}

for key, info_path in info_files.items():
    with open(info_path, "w") as info_file:
        json.dump(dataset_info, info_file)
