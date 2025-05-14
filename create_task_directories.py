import os
import json
from pathlib import Path

# 确保目录存在
task_dirs = ["media/task_1", "media/task_2"]
for task_dir in task_dirs:
    path = Path(task_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    # 创建必要的JSON文件
    files = ["data.json", "tag.json", "label.json", "relation.json"]
    for file in files:
        file_path = path / file
        if not file_path.exists():
            file_path.write_text(json.dumps([]), encoding='utf-8')
            print(f"创建文件: {file_path}")

print("任务目录和文件创建完成！")