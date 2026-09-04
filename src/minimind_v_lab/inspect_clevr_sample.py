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

SPATIAL_RELATIONS = {
    "left",
    "right",
    "front",
    "behind",
    "above",
    "below",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

def is_spatial_question(question: dict) -> bool:
    for step in question["program"]:
        if step["function"] != "relate":
            continue

        relation = step["value_inputs"][0]#根据value_inputs获取关系类型

        if relation in SPATIAL_RELATIONS:
            return True

    return False
def get_spatial_relations(question: dict) -> list[str]:
    relations = []

    for step in question["program"]:
        if step["function"] == "relate":
            relations.extend(step["value_inputs"])

    return relations#将所有的关系类型都返回为列表

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

    questions = questions_data["questions"]
    scenes = scenes_data["scenes"]

    question = questions[0]

    print("question index:", question["question_index"])
    print("image index:", question["image_index"])
    print("question text:", question["question"])
    print("answer:", question["answer"])

    print("program:")

    for step in question["program"]:
        print("  ", step)

    scene_by_image_index = {
        scene["image_index"]: scene
        for scene in scenes
    }
    scene = scene_by_image_index[question["image_index"]]

    print("scene image filename:", scene["image_filename"])
    print("object count:", len(scene["objects"]))

    first_object = scene["objects"][0]

    print("first object:")
    print(first_object)

    print("relationship types:")

    for relation_name, relation_data in scene["relationships"].items():
        print(
            relation_name,
            "rows:",
            len(relation_data),
        )

    spatial_questions = [
        question
        for question in questions
        if is_spatial_question(question)
    ] 
    print(
        "spatial question count:",
        len(spatial_questions),
    )

    first_spatial_question = spatial_questions[0]

    print(
        "first spatial question:",
        first_spatial_question["question"],
    )

    print(
        "spatial relations:",
        get_spatial_relations(first_spatial_question),
    )

    print(
        "program length:",
        len(first_spatial_question["program"]),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())