from __future__ import annotations

import json
import os
from pathlib import Path


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
#定位到数据集

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> int:
    questions_data = load_json(QUESTIONS_PATH)
    scenes_data = load_json(SCENES_PATH)

    print("question keys:", questions_data.keys())
    print("scene keys:", scenes_data.keys())

    print(
        "question count:",
        len(questions_data["questions"]),
    )

    print(
        "scene count:",
        len(scenes_data["scenes"]),
    )
    #统计总个数
    return 0


if __name__ == "__main__":
    raise SystemExit(main())