import os
import sys
from base64 import b64encode
from io import BytesIO

import ollama
from PIL import Image


def load_image(image_path: str) -> Image.Image:
    image = Image.open(image_path)
    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")
    return image


def encode_image(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format=image.format)
    return b64encode(buffered.getvalue()).decode("utf-8")


def generate_filename_from_answer(answer: str) -> str:
    # Remove file extensions if present
    answer = answer.rsplit(".", 1)[0]

    filename = "".join(char.lower() if char.isalnum() else "_" for char in answer)
    words = filter(None, filename.split("_"))
    filtered_words = []
    for word in words:
        if not word.isdigit():
            filtered_words.append(word)

    if not filtered_words:
        return "unnamed_image"  # Fallback if all words are numbers

    return "_".join(filtered_words)


def generate_filename(image_path: str) -> str:
    image = load_image(image_path)
    encoded_image = encode_image(image)
    response = ollama.chat(
        model="moondream",
        messages=[
            {
                "role": "user",
                "content": (
                    "Analyze the image and generate a concise filename (3-5 words max) "
                    "that captures its essence. Consider the following aspects:\n"
                    "1. Primary subject and action\n"
                    "2. Key visual characteristics (e.g., color, composition)\n"
                    "3. Setting or context\n"
                    "4. Unique or distinctive elements\n"
                    "5. Mood or theme (if prominent)\n"
                    "Use lowercase words separated by underscores, avoiding generic terms. "
                    "The filename should enable easy identification within a large collection. "
                    "Do not use numbers or dates as the sole description."
                ),
                "images": [encoded_image],
            }
        ],
    )
    answer = response["message"]["content"]
    return generate_filename_from_answer(answer)


def get_new_path(image_path: str, filename: str) -> str:
    directory = os.path.dirname(image_path)
    base_path = os.path.join(directory, filename)
    extension = os.path.splitext(image_path)[1]
    new_path = base_path + extension
    counter = 1
    while os.path.exists(new_path):
        new_path = f"{base_path}_{counter}{extension}"
        counter += 1
    return new_path


def rename_image(image_path: str, new_path: str) -> None:
    try:
        os.rename(image_path, new_path)
    except Exception as e:
        print(f"Error renaming image {image_path} to {new_path}: {e}")


def process_image(image_path: str, index: int, total: int) -> None:
    try:
        filename = generate_filename(image_path)
        new_path = get_new_path(image_path, filename)
        if new_path != image_path:
            rename_image(image_path, new_path)
            print(f"{index}/{total} Image renamed to: {new_path}")
        else:
            print(f"{index}/{total} Image already has the suggested name: {image_path}")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")


def process_folder(folder_path: str) -> None:
    supported_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    image_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_path)
        for file in files
        if file.lower().endswith(supported_extensions)
    ]

    image_count = len(image_paths)
    print(f"Found {image_count} image{'s' if image_count != 1 else ''} in the folder.")

    for index, image_path in enumerate(image_paths, start=1):
        process_image(image_path, index, image_count)


def main(path: str) -> None:
    if os.path.isfile(path):
        process_image(path, 1, 1)
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
    main(path)
