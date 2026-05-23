import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("baseline_model.keras")
CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]  # sửa lại đúng tên class của bạn
IMG_SIZE = 150

def predict(img):
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
    description="Upload ảnh để phân loại cảnh quan"
)

demo.launch()
