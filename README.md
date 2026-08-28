# Recyclable Waste Classifier

An image classifier that predicts which recycling category an item belongs to —
cardboard, glass, metal, paper, plastic or trash — using transfer learning on
MobileNetV2, exported to ONNX for a lightweight, GPU-free deployment.

- **Base model**: MobileNetV2 (pretrained on ImageNet), fine-tuned on
  [`garythung/trashnet`](https://huggingface.co/datasets/garythung/trashnet)
- **Training**: PyTorch + torchvision (local/dev only)
- **Inference**: ONNX Runtime (no PyTorch needed at deploy time — same lesson learned
  from the [RAG chatbot project](https://github.com/NicoDevGod/rag-chatbot-cv): keep
  the deployed app's memory footprint small)
- **UI**: [Gradio](https://www.gradio.app/)

## Local setup

Training and inference use **different** dependencies — training needs PyTorch
(heavy, only for producing the model), the deployed app only needs ONNX Runtime
(light).

### 1. Train the model

```bash
python -m venv .venv-train
.venv-train\Scripts\activate   # Windows
pip install -r requirements-train.txt
python train.py
```

This streams a balanced sample (150 images/class) from the dataset — no need to
download the full ~3.7GB dataset — fine-tunes MobileNetV2 for 5 epochs, and writes
`model/waste_classifier.onnx` + `model/labels.json`.

### 2. Run the app

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Gradio prints a local URL (usually http://127.0.0.1:7860). Upload a photo and it
predicts the category.

## Deploying to Render

This repo includes a [`render.yaml`](render.yaml) Blueprint:

1. Sign in at https://dashboard.render.com.
2. **New → Blueprint** → pick this repo. No environment variables are required
   (unlike the RAG chatbot, there's no external API — everything runs locally).
3. Deploy. `model/waste_classifier.onnx` is committed to the repo, so no training
   step runs in production.
