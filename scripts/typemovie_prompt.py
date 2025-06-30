import json

# 读取txt文件内容
txt_file_path = r'D:\common_tools\IBench\data\prompts\human_action_longer.txt'
with open(txt_file_path, 'r', encoding='utf-8') as file:
    txt_content = file.read()

# 将txt内容分割成不同的行
entries = txt_content.strip().split('\n')

# 初始化JSON数据结构
json_data = {"datas": []}

# 处理每个条目并添加到JSON数据结构
for i, entry in enumerate(entries):
    json_data["datas"].append({
        "num": i,
        "prompt": entry,
        "prompt_attr": ["long"],
        "prompt_style": ["human_action_longer"],
    })

# 保存为JSON文件
json_file_path = '../data/prompts/human_action_longer_template.json'
with open(json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(json_data, json_file, ensure_ascii=False, indent=2)

print(f'转换完成，结果保存在 {json_file_path}')