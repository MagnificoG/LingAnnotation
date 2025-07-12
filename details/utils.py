# encoding: utf-8

"""
处理数据的其他函数
"""
from . import config
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_txt(file) -> List[Dict]:
    lines: list[str] = file.read().decode('utf-8').splitlines()
    # 创建数据
    data = [
        {
            config.TEXT: line.strip(),
            config.TAGS: [],
            config.LABELS: [],
            config.RELATIONS: []
        } for line in lines if line.strip()
    ]
    return data

def parse_tabular_file(file) -> str:
    """Parse tabular data from a file into a JSON string."""
    suffix = Path(file.name).suffix.lower()
    
    if suffix == '.csv':
        df = pd.read_csv(file)
    elif suffix in ['.xls', '.xlsx']:
        df = pd.read_excel(file)
        logging.info(f"Excel file parsed with {len(df)} rows and {len(df.columns)} columns.")
        logging.info(f"First 5 rows:\n{df.head()}")
    elif suffix == '.json':
        df = pd.read_json(file)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    
    records = df.to_json(orient='records')
    # parsed_json = json.loads(records)
    
    return records

def parse_file(file) -> List[Dict]:
    """根据文件后缀名解析数据

    Args:
        file: 文件流

    Raises:
        ValueError: 不支持的文件类型
    """
    suffix = Path(file.name).suffix.lower()
    if suffix == '.txt':
        return parse_txt(file)
    elif suffix in ['.csv', '.xls', '.xlsx', '.json']:
        # This will now return a JSON string for tabular data,
        # which is not ideal for the old upload_data view.
        # This discrepancy will be resolved when we implement the new views.
        return parse_tabular_file(file)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")