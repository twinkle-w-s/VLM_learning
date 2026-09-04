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
    print("columns:", table.num_columns)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())