import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_DIR = Path(__file__).parent / "model"
IMAGE_SIZE = 160
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

LABELS_ES = {
    "cardboard": "cartón",
    "glass": "vidrio",
    "metal": "metal",
    "paper": "papel",
    "plastic": "plástico",
    "trash": "basura",
}


def load_model():
    session = ort.InferenceSession(str(MODEL_DIR / "waste_classifier.onnx"))
    labels = json.loads((MODEL_DIR / "labels.json").read_text())
    return session, labels


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    array = array.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(array, axis=0)


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()


def make_predict_fn(session, labels):
    input_name = session.get_inputs()[0].name

    def predict(image: Image.Image):
        if image is None:
            return {}
        inputs = preprocess(image)
        (logits,) = session.run(None, {input_name: inputs})
        probs = softmax(logits[0])
        return {LABELS_ES.get(label, label): float(prob) for label, prob in zip(labels, probs)}

    return predict


def main():
    session, labels = load_model()
    demo = gr.Interface(
        fn=make_predict_fn(session, labels),
        inputs=gr.Image(type="pil", label="Sube una foto del objeto"),
        outputs=gr.Label(num_top_classes=6, label="Categoría predicha"),
        title="Clasificador de Residuos Reciclables",
        description=(
            "Sube una foto de un objeto y el modelo predice a qué categoría de "
            "reciclaje pertenece: cartón, vidrio, metal, papel, plástico o basura. "
            "MobileNetV2 ajustado (fine-tuned), corriendo en ONNX Runtime (sin GPU)."
        ),
    )
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    main()
