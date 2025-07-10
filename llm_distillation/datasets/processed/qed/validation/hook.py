import pyarrow as pa
import pyarrow.ipc as ipc

# 读取Arrow文件
file_path = 'data-00000-of-00001.arrow'
with open(file_path, 'rb') as f:
    reader = ipc.open_stream(f)
    table = reader.read_all()

# 将Arrow Table转换为Pandas DataFrame
df = table.to_pandas()
# 打印原始数据的前几行以检查结构
print(df.head().original_nq_answers)
# 假设你已知的值是known_answer
known_answer = 'Christina Perri'


# 定义一个函数来检查每一行的结构并过滤包含已知值的行
def contains_known_answer(answers, known_answer):
    for sublist in answers:
        if isinstance(sublist, dict) and 'string' in sublist and known_answer in sublist['string']:
            return True
    return False

# 筛选包含已知值的行
filtered_rows = df[df['original_nq_answers'].apply(lambda x: contains_known_answer(x, known_answer))]

# 打印过滤后的结果
print("Filtered DataFrame:")
print(filtered_rows)

# 提取question和paragraph_text字段
if not filtered_rows.empty:
    question = filtered_rows['question'].values[0]
    paragraph_text = filtered_rows['paragraph_text'].values[0]
    print(f"Question: {question}")
    print(f"Paragraph Text: {paragraph_text}")
else:
    print("No matching row found.")