import json
import os

input_file_path = "/Users/zhengjiayi/PythonProjects/mend_qwen/my_version/diverse_augmented.jsonl"
output_file_path = "/Users/zhengjiayi/PythonProjects/mend_qwen/my_version/converted_diverse_augmented.jsonl"

converted_entries = []

# Ensure the directory for the output file exists
output_dir = os.path.dirname(output_file_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(input_file_path, 'r', encoding='utf-8') as infile:
    for line in infile:
        try:
            original_entry = json.loads(line.strip())

            new_entry = {
                "input": original_entry.get("edit_prompt", ""),
                "prediction": original_entry.get("edit_target", ""),
                "alternatives": [], # 默认为空列表，你可以根据实际需求修改
                "filtered_rephrases": [original_entry.get("rephrase_prompt", "")],
                "output": [{"answer": original_entry.get("edit_target", "")}]
            }
            converted_entries.append(new_entry)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e} in line: {line.strip()}")
            continue
        except KeyError as e:
            print(f"Missing key: {e} in entry: {original_entry}")
            continue

with open(output_file_path, 'w', encoding='utf-8') as outfile:
    for entry in converted_entries:
        outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"数据转换完成。转换后的文件保存在：{output_file_path}")
