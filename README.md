# Hinglish Toxicity Detector

A deep learning web app that detects offensive content in Hinglish (Hindi-English) text. Fine-tuned Phi-3.5-mini using QLoRA on 30K Hinglish comments, and deployed as a Gradio app on Hugging Face Spaces.

**Live Demo:** [huggingface.co/spaces/armaan-vala/hinglish-toxicity-detector](https://huggingface.co/spaces/armaan-vala/hinglish-toxicity-detector)

## Dataset

- **30,452 Hinglish comments** — 16K not offensive + 14K offensive (balanced)
- Cleaned from raw dataset: removed duplicates, short/long texts, invalid labels
- Source: Hinglish toxic comment dataset from Kaggle

## Model

| Detail | Value |
|---|---|
| Base Model | Phi-3.5-mini-instruct (3.8B params) |
| Fine-tuning | QLoRA (4-bit quantization + LoRA adapters) |
| Trainable Params | 12.5M / 3.8B (0.33%) |
| Training | 1 epoch, batch size 32, lr 2e-4, fp16 |
| GPU | Google Colab T4 (free tier) |

Merged model hosted on HF Hub: [armaan-vala/hinglish-toxic-merged](https://huggingface.co/armaan-vala/hinglish-toxic-merged)

## Tech Stack

- **Transformers + PEFT** — model loading and LoRA fine-tuning
- **Unsloth** — optimized QLoRA training on Colab
- **bitsandbytes** — 4-bit quantization
- **FastAPI + Uvicorn** — local web app
- **Gradio** — HF Spaces deployment
- **Google Colab** — training environment

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/armaan-vala/hinglish-toxicity-detector.git
cd hinglish-toxicity-detector

# Install dependencies
pip install -r requirements.txt

# Download adapter weights (place adapter_model.safetensors in model/ folder)
# Get it from the training notebook output or HF Hub

# Run the web app
uvicorn app:app --reload

# Open in browser
# http://localhost:8000
```

## Project Structure

```
├── app.py                  # FastAPI local web app
├── training_colab.ipynb    # Full training notebook (run on Colab)       
├── requirements.txt        # Local dependencies
├── hinglish_toxic_clean.csv # Cleaned dataset
├── model/                  # LoRA adapter files (tokenizer + config)
└── deploy/
    ├── app.py              # Gradio app for HF Spaces
    ├── requirements.txt    # Space dependencies
    └── README.md           # HF Space metadata
```

## Author

**Armaan Vala** — [GitHub](https://github.com/armaan-vala)
