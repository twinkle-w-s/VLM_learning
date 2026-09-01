import json
import os
from pathlib import Path

def get_storage_path():
    return (
        Path("/data")
        / os.environ["USER"]
        / "vlm_learning"
    )
def get_data_root():
    storage_root = get_storage_path()
    return storage_root / "data"

def find_clevr_directories(data_root):
    candidates = []

    for path in data_root.rglob("*"):
        if not path.is_dir():
            continue

        directory_name = path.name.lower()

        if "clevr" in directory_name:
            candidates.append(path)

    return candidates
def list_dataset_files(data_root):
    json_files = sorted(data_root.rglob("*.json"))
    image_files = sorted(data_root.rglob("*.png"))

    print("JSON files:")

    for path in json_files[:20]:
        print("-", path)

    print("PNG image count:", len(image_files))

    return json_files, image_files
def inspect_json_file(json_path):
    print("Inspecting JSON:", json_path)

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    print("top-level type:", type(data).__name__)

    if isinstance(data, dict):
        print("top-level keys:", list(data.keys()))

    elif isinstance(data, list):
        print("list length:", len(data))

        if len(data) > 0:
            print(
                "first item type:",
                type(data[0]).__name__,
            )

            if isinstance(data[0], dict):
                print(
                    "first item keys:",
                    list(data[0].keys()),
                )

    return data
def choose_json_files(json_files):
    question_files = []
    scene_files = []

    for path in json_files:
        name = path.name.lower()

        if "question" in name:
            question_files.append(path)

        if "scene" in name:
            scene_files.append(path)

    return question_files, scene_files
def main():
    data_root = get_data_root()

    print("data root:", data_root)

    if not data_root.exists():
        raise FileNotFoundError(
            f"Data directory was not found: {data_root}"
        )

    clevr_directories = find_clevr_directories(
        data_root
    )

    print("CLEVR directories:")

    for path in clevr_directories:
        print("-", path)

    json_files, image_files = list_dataset_files(
        data_root
    )

    question_files, scene_files = choose_json_files(
        json_files
    )

    print("question files:")

    for path in question_files:
        print("-", path)

    print("scene files:")

    for path in scene_files:
        print("-", path)
        
    if question_files:
        question_data = inspect_json_file(
            question_files[0]
        )

    if scene_files:
        scene_data = inspect_json_file(
            scene_files[0]
        )

        if isinstance(scene_data, dict):
            scenes = scene_data.get("scenes", [])

            if scenes:
                print(
                    "first scene keys:",
                    list(scenes[0].keys()),
                )
if __name__ == "__main__":
    main()