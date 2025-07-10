import pyarrow as pa
import os

file_path = 'data-00000-of-00001.arrow'
print(f"File size: {os.path.getsize(file_path)} bytes")

# 尝试以文件方式读取
try:
    with pa.memory_map(file_path, 'r') as source:
        table = pa.ipc.open_file(source).read_all()
    print("读取成功，文件方式（File）")
except pa.lib.ArrowInvalid:
    print("尝试以文件方式（File）读取失败，尝试以流方式（Stream）")

    try:
        with pa.memory_map(file_path, 'r') as source:
            table = pa.ipc.open_stream(source).read_all()
        print("读取成功，流方式（Stream）")
    except Exception as e:
        print(f"读取文件时出现错误：{e}")

# 如果读取成功，转换为 Pandas DataFrame
try:
    df = table.to_pandas()
    print(df.head(2).summary_llama)
    print(df.head(2).answers)
except NameError:
    print("表格未成功加载，无法转换为 DataFrame")
