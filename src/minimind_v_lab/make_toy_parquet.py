from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

def read_image_bytes(image_path: Path) -> bytes:
    with Image.open(image_path) as image:
        image=image.convert("RGB")

        buffer= io.BytesIO()#创建内存缓冲
        image.save(buffer, format="JPNG",quality=95)#图片写入缓冲，以jepg编码

        return buffer.getvalue()



def build_row(image_bytes: bytes, question: str) -> dict:
    conversations = [
        {
            "role": "user",
            "content": f"{question}<image>",
        },
        {
            "role": "assistant",
            "content": answer,
        },
    ]#定义问答形式
    return {
        "image_bytes": image_bytes,
        "conversations": json.dumps(conversations, ensure_ascii=False),
    }#把问答按照json格式存储在字典中，image_bytes是图片的二进制数据


def main()->int:#函数返回int
    parser = argparse.ArgumentParser(
        description="Create a minimal MiniMind-V Parquet dataset."

    )
    parser.add_argument("image_path", type=Path)
    parser.add_argument("output_path", type=Path)

    args = parser.parse_args()
    image_bytes = read_image_bytes(args.image_path)
    rows = [
        build_row(
            image_bytes=image_bytes,
            question="请描述这张图片中的主要物体。",
            answer="图片中有一个主要物体。",
        ),
        build_row(
            image_bytes=image_bytes,
            question="请简要说明这张图片的内容。",
            answer="这是一张包含主要物体的图片。",
        ),
    ]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, args.output_path)

    print(f"wrote: {args.output_path}")
    print(f"rows: {table.num_rows}")
    print(f"columns: {table.column_names}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())