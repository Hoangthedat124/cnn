import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download
import os

MODEL_PATH = "baseline_model.keras"
if not os.path.exists(MODEL_PATH):
    print("Đang tải model...")
    MODEL_PATH = hf_hub_download(
        repo_id="Dastsimcu/landscape-cnn",
        filename="baseline_model.keras",
        repo_type="model"
    )

model = load_model(MODEL_PATH)

# Sửa lại đúng tên class của bạn (theo alphabet)
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
IMG_SIZE = 150

def predict(img):
    from PIL import Image
    import numpy as np
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)[0]
    return {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Phân loại cảnh quan CNN",
    description="Upload ảnh để check class"
)

demo.launch()
