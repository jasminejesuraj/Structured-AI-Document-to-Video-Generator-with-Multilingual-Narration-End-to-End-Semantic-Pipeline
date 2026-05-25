# AI Document To Video Generator

An AI-powered automated multimedia generation pipeline that converts structured PDF/DOCX documents into educational videos using semantic parsing, LLM-based narration generation, Stable Diffusion image synthesis, SVD-based video generation, multilingual TTS narration, and automated video assembly.

# Pipeline Overview

The system transforms documents into videos through multiple AI stages:

PDF/DOCX → Semantic Parsing → Storyboard Generation → Image Generation → Cloud Upload → Colab Video Assembly → Final Video

---

# Features

- Structured document parsing
- Semantic chunking and section extraction
- LLM-based summarization and narration generation
- Scene-wise storyboard creation
- AI image generation using SDXL Turbo
- Automated JSON-based pipeline tracking
- Google Drive upload integration
- SVD-based video generation for motion/animation synthesis
- Colab-based Text-to-Speech (TTS) narration generation
- Automated video assembly pipeline

---

# Project Structure

```text
Structured-AI-Document-to-Video/
│
├── input/
│   └── README (instructions for adding input_document here)
│
├── sample_outputs/
│   ├── sample_scenes.png
│   └── sample_video.mp4
│
├── main.py
├── prompt_generation.py
├── image.py
├── upload_to_drive.py
├── sample_storyboard.json
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/jasminejesuraj/Structured-AI-Document-to-Video-Generator-with-Multilingual-Narration-End-to-End-Semantic-Pipeline.git
cd AI-Document-To-Video
```

---

## Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

# External Requirements

## Ollama

Install Ollama from:

https://ollama.com/


Optional: Pull the model manually

Only run this if the model is not already available locally:

```bash
ollama pull llama3.2:3b
```

Note
*If the model is not present, it will be automatically downloaded when required. However, Ollama must be installed first, as it manages both model execution and downloads.*

---

## Google OAuth Credentials

Place your Google OAuth credentials file in the project root:

```text
client_secrets.json
```

This file is required for Google Drive upload functionality.

---

# Usage

## Step 1 — Add Input Documents

Place PDF or DOCX files inside:

```text
input/
```

---

## Step 2 — Start Main Pipeline

Run:

```bash
python main.py
```

This automatically performs:

1. Prompt and storyboard generation
2. AI image generation
3. Google Drive upload

---

# Pipeline Stages

## Stage 1 — Storyboard Generation

Handled by:

```text
prompt_generation.py
```

Functions:
- Structural document parsing
- Semantic chunking
- LLM summarization
- Scene narration generation
- Visual prompt creation
- Storyboard JSON generation

Generated file:

```text
storyboard.json
```

---

## Stage 2 — Image Generation

Handled by:

```text
image.py
```

Functions:
- Reads storyboard.json
- Generates AI images using SDXL Turbo
- Updates storyboard with image paths

Generated outputs:

```text
outputs/scene_1.png
outputs/scene_2.png
...
```

---

## Stage 3 — Google Drive Upload

Handled by:

```text
upload_to_drive.py
```

Functions:
- Creates project folders in Google Drive
- Uploads storyboard.json
- Uploads generated images

---

# Colab Video Generation Phase

After running `main.py`, continue the pipeline in Google Colab.

The Colab stage performs:
- multilingual narration generation
- text-to-speech synthesis
- scene video generation
- audiovisual synchronization
- final video assembly

---

# Open Colab Notebook

[Open Colab Notebook](https://colab.research.google.com/drive/1E6a-ueckrywBs4tMq1IE0RKgRfCvwXqw#scrollTo=G1pxLhrlBZDp)

---

# Example Output

The pipeline generates:
- scene-wise storyboard JSON
- AI-generated scene images
- multilingual narration
- final educational video

---

# Notes

- Generated files inside `outputs/` are ignored by GitHub
- `storyboard.json` is automatically generated during runtime
- Do NOT upload `client_secrets.json` publicly

---

# Technologies Used

- Python
- Ollama
- LLaMA 3.2
- Stable Diffusion XL Turbo
- Diffusers
- PyTorch
- PyMuPDF
- LlamaIndex
- Google Drive API
- Google Colab

---

# Future Improvements

- Avatar-based narration
- Real-time streaming generation
- Web-based deployment
- Interactive scene editing
- Advanced multilingual support

---

# License

This project is licensed under the MIT License.

It was primarily developed for educational and research purposes.
