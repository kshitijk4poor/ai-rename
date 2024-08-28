import os
import sys
from typing import Any, Tuple, List
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer  # ignore

model_id: str = "vikhyatk/moondream2"
revision: str = "2024-08-26"
model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path)


def encode_image(image: Image.Image) -> Tuple[Any, Any]:
    return model.encode_image(image)


def generate_filename_from_answer(answer: str) -> str:
    filename = "".join(char.lower() if char.isalnum() else "_" for char in answer)
    return "_".join(filter(None, filename.split("_")))


def generate_filename(image_path: str) -> str:
    image = load_image(image_path)
    enc_image = encode_image(image)
    answer = model.answer_question(
        enc_image,
        "Generate a lowercase filename with no extension, no special "
        "characters, only key elements, one word if possible, in noun-verb "
        "format. Avoid generic terms and aim for specificity. The filename "
        "should enable easy identification and organization of the image "
        "within a large collection. The filename should be concise yet "
        "informative, using 3-5 words separated by underscores if needed.",
        tokenizer,
    )
    return generate_filename_from_answer(answer)


def get_new_path(image_path: str, filename: str) -> str:
    directory = os.path.dirname(image_path)
    return os.path.join(directory, filename + os.path.splitext(image_path)[1])


def rename_image(image_path: str, new_path: str) -> None:
    os.rename(image_path, new_path)


def process_image(image_path: str) -> None:
    filename = generate_filename(image_path)
    new_path = get_new_path(image_path, filename)
    rename_image(image_path, new_path)
    print(f"Image renamed to: {new_path}")


def process_folder(folder_path: str) -> None:
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(supported_extensions):
                image_path = os.path.join(root, file)
                process_image(image_path)


def main(path: str) -> None:
    if os.path.isfile(path):
        process_image(path)
    elif os.path.isdir(path):
        process_folder(path)
    else:
        print(f"Invalid path: {path}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <image_path_or_folder_path>")
        sys.exit(1)

    path = sys.argv[1]
    try:
        main(path)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
