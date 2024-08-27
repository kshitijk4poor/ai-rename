import sys
import os
from typing import Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

model_id: str = "vikhyatk/moondream2"
revision: str = "2024-08-26"
model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)


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


def main(image_path: str) -> None:
    filename = generate_filename(image_path)
    new_path = get_new_path(image_path, filename)
    rename_image(image_path, new_path)
    print(f"Image renamed to: {new_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    try:
        main(image_path)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
