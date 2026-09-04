import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image



def summarize_row(row: dict[str, any]) -> dict:
    conversations= json.loads(row["conversations"])
    roles=[turn["role"] for turn in conversations]

    image_bytes=row["image_bytes"]

    if isinstance(image_bytes, list):
        image_count=len(image_bytes)
    else:
        image_count=1

    return {
        
        "roles": roles,
        "image_count": image_count,
        "conversation_count": len(conversations)
    }

def main()->int:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_path", type=Path)
    
    args = parser.parse_args()

    table = pq.read_table(args.parquet_path)
    

    print("rows:", table.num_rows)
    print("columns:", table.column_names)

    row=table.slice(0,1).to_pylist()[0]#table.slice(0,1)取第一行，返回一个新的表，包含从索引0开始的1行数据。
    #to_pylist()将表转换为Python列表，其中每个元素是一个字典，表示一行数据。然后[0]获取列表中的第一个元素，即第一行数据。
    summary = summarize_row(row)
    print("summary:", summary)

    
    print("conversations type:", type(row["conversations"]).__name__)#打印出字段的类型名
    print("conversations raw:", row["conversations"])

    print("image_bytes type:", type(row["image_bytes"]).__name__)
    print("image_bytes length:", len(row["image_bytes"]))

    #查看对话内容
    conversations= json.loads(row["conversations"])

    for turn in conversations:
        role=turn["role"]
        content=turn["content"]
        print(f"{role}: {content}")
    #继续查看

    image_bytes = row["image_bytes"]

    if isinstance(image_bytes, list):
        image_bytes = image_bytes[0]

    with Image.open(io.BytesIO(image_bytes)) as image:
        print("image format:", image.format)
        print("image size:", image.size)
        print("image mode:", image.mode)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())