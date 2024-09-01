# 🖼️ AI Image Renamer

## 📝 Description

AI Image Renamer is a Python script that uses AI to automatically generate descriptive and meaningful filenames for images. It analyzes the content of each image and creates a concise, relevant filename based on the image's key characteristics.

## 🚀 Features

- 🧠 Uses AI to analyze image content
- 🏷️ Generates concise, descriptive filenames (3-5 words)
- 📁 Supports processing individual images or entire folders
- 🔄 Handles filename conflicts automatically
- 🖼️ Supports various image formats (JPG, PNG, BMP, GIF)

## 📋 TODO

- [ ] Add more test cases and edge case handling
- [x] Switch to ollama for inference
- [x] Enhance the rename conflict resolution algorithm
- [ ] Expand supported image formats

## 🛠️ Installation

1. Download and install Ollama from [ollama.com](https://ollama.com/download).
2. Clone this repository:
   ```
   git clone https://github.com/kshitijk4poor/ai-rename.git
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ollama pull moondream
   ```

## 🔧 Usage

Run the script with a path to an image or a folder containing images:
```
python test.py <image_path_or_folder_path>
```

## 👏 Acknowledgements

- [Moondream](https://github.com/vikhyat/moondream) by Vikhyat Korrapati for the AI model