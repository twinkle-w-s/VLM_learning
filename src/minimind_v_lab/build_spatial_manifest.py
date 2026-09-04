from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from inspect_clevr_sample import (
    get_spatial_relations,
    is_spatial_question,
    load_json,
)

DATA_ROOT = (
    Path("/data")
    / os.environ["USER"]
    / "vlm_learning"
    / "CLEVR_v1.0"
)

QUESTIONS_PATH = (
    DATA_ROOT
    / "questions"
    / "CLEVR_train_questions.json"
)

SCENES_PATH = (
    DATA_ROOT
    / "scenes"
    / "CLEVR_train_scenes.json"
)

def build_record(
    question: dict,
    scene: dict,
) -> dict:
    return {
        "question_index": question["question_index"],
        "image_index": question["image_index"],
        "image_filename": scene["image_filename"],
        "question": question["question"],
        "answer": question["answer"],
        "program": question["program"],
        "spatial_relations": get_spatial_relations(question),
        "program_length": len(question["program"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--limit", type=int, default=1000)

    args = parser.parse_args()

    questions_data = load_json(QUESTIONS_PATH)
    scenes_data = load_json(SCENES_PATH)

    questions = questions_data["questions"]
    scenes = scenes_data["scenes"]

    scene_by_image_index = {
        scene["image_index"]: scene
        for scene in scenes
    }#建立场景索引，把场景索引设置为image-idx

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    written = 0

    with args.output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for question in questions:
            if written >= args.limit:
                break

            if not is_spatial_question(question):
                continue

            scene = scene_by_image_index[
                question["image_index"]
            ]

            record = build_record(question, scene)

            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            written += 1

    print("written:", written)
    print("output:", args.output_path)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())