# 这个脚本串联全量 CLEVR 空间数据的生成、标注、图像级切分和初始配方。

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(label: str, command: list[str]) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    manifest_dir = args.output_root / "manifests"
    curated_dir = args.output_root / "curated"

    spatial = manifest_dir / "spatial_all.jsonl"
    annotated = manifest_dir / "spatial_all_annotated.jsonl"
    train = curated_dir / "spatial_train.jsonl"
    val = curated_dir / "spatial_val.jsonl"
    recipe_train = curated_dir / "spatial_initial_recipe_train.jsonl"
    val_parquet = curated_dir / "spatial_val.parquet"
    recipe_parquet = curated_dir / "spatial_initial_recipe_train.parquet"

    python = sys.executable
    run_step(
        "build full spatial manifest",
        [
            python,
            str(project_root / "src/minimind_v_lab/manifest/build_spatial_manifest.py"),
            str(spatial),
            "--limit",
            str(args.limit),
        ],
    )
    run_step(
        "annotate manifest",
        [
            python,
            str(project_root / "src/minimind_v_lab/manifest/annotate_manifest.py"),
            str(spatial),
            str(annotated),
        ],
    )
    run_step(
        "split by image",
        [
            python,
            str(project_root / "src/minimind_v_lab/manifest/split_by_image.py"),
            str(annotated),
            str(train),
            str(val),
            "--val-ratio",
            str(args.val_ratio),
            "--seed",
            str(args.seed),
        ],
    )
    for split_name, split_path in [
        ("train", train),
        ("validation", val),
    ]:
        run_step(
            f"validate {split_name} manifest",
            [
                python,
                str(project_root / "src/minimind_v_lab/validate/validate_manifest.py"),
                str(split_path),
            ],
        )
    run_step(
        "build initial recipe",
        [
            python,
            str(project_root / "src/minimind_v_lab/manifest/build_initial_recipe.py"),
            str(train),
            str(recipe_train),
            "--seed",
            str(args.seed),
        ],
    )
    run_step(
        "summarize initial recipe",
        [
            python,
            str(project_root / "src/minimind_v_lab/manifest/summarize_manifest.py"),
            str(recipe_train),
        ],
    )
    run_step(
        "convert validation parquet",
        [
            python,
            str(project_root / "src/minimind_v_lab/convert/convert_manifest_to_parquet_streaming.py"),
            str(val),
            str(val_parquet),
        ],
    )
    run_step(
        "convert recipe parquet",
        [
            python,
            str(project_root / "src/minimind_v_lab/convert/convert_manifest_to_parquet_streaming.py"),
            str(recipe_train),
            str(recipe_parquet),
        ],
    )
    for parquet_name, parquet_path in [
        ("validation", val_parquet),
        ("recipe", recipe_parquet),
    ]:
        run_step(
            f"validate {parquet_name} parquet",
            [
                python,
                str(project_root / "src/minimind_v_lab/validate/validate_schema.py"),
                str(parquet_path),
                "--max-rows",
                "100",
            ],
        )

    print("\nprepared files:")
    for path in [
        spatial,
        annotated,
        train,
        val,
        recipe_train,
        val_parquet,
        recipe_parquet,
    ]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
