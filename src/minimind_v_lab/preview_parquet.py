import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

def main()->int:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_path", type=Path)
    
    args = parser.parse_args()

    table = pq.read_table(args.parquet_path)
    

    print("rows:", table.num_rows)
    print("columns:", table.column_names)

    row=table.slice(0,1).to_pydict()[0]#table.slice(0,1)取第一行，返回一个新的表，包含从索引0开始的1行数据。
    #to_pydict()将表转换为Python字典，其中键是列名，值是列数据的列表。然后[0]获取字典中第一列的数据列表的第一个元素，即第一行数据。

    print("conversations type:", type(row["conversations"]).__name__)#打印出字段的类型名
    print("conversations raw:", row["conversations"])

    print("image_bytes type:", type(row["image_bytes"]).__name__)
    print("image_bytes length:", len(row["image_bytes"]))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())